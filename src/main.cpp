#include <algorithm>
#include <array>
#include <assert.h>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN        // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE        // Vulkan clip space depth is [0, 1]
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "camera.hpp"
#include "ply_loader.hpp"

#include <string>

constexpr uint32_t WIDTH                = 800;
constexpr uint32_t HEIGHT               = 600;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

// Camera data for the splat shader. Layout matches CameraUBO in shaders/shader.slang.
struct UniformBufferObject
{
	glm::mat4 view;
	glm::mat4 proj;
	glm::vec2 viewport;        // framebuffer size in pixels
	glm::vec2 focal;           // fx, fy in pixels
};

// Selects the back-to-front sort strategy; toggle at runtime with G.
enum class SortMode
{
	Cpu,        // std::sort on the host, memcpy into the index buffer
	Gpu         // compute bitonic sort, in place in the index buffer
};

// Must match SortPush in shaders/compute.slang.
struct SortPushConstants
{
	glm::vec4 viewRow2;        // z = dot(viewRow2.xyz, pos) + viewRow2.w
	uint32_t  splatCount;
	uint32_t  paddedCount;
	uint32_t  j;               // bitonic partner stride
	uint32_t  k;               // bitonic subsequence size
};

class Application
{
  public:
	explicit Application(std::string plyPath) : plyPath(std::move(plyPath)) {}

	void run()
	{
		loadScene();
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	std::string                      plyPath;
	Scene                            scene;
	uint32_t                         splatCount  = 0;
	uint32_t                         paddedCount = 0;        // splatCount rounded up to a power of two (bitonic sort)
	Camera                           camera;

	GLFWwindow                      *window = nullptr;
	vk::raii::Context                context;
	vk::raii::Instance               instance       = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::SurfaceKHR             surface        = nullptr;
	vk::raii::PhysicalDevice         physicalDevice = nullptr;
	vk::raii::Device                 device         = nullptr;
	uint32_t                         queueIndex     = ~0;
	vk::raii::Queue                  queue          = nullptr;
	vk::raii::SwapchainKHR           swapChain      = nullptr;
	std::vector<vk::Image>           swapChainImages;
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;
	vk::Extent2D                     swapChainExtent;
	std::vector<vk::raii::ImageView> swapChainImageViews;

	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      pipelineLayout      = nullptr;
	vk::raii::Pipeline            graphicsPipeline    = nullptr;

	// Compute sort: one set-0 layout shared by both pipelines (init + bitonic pass).
	SortMode                             sortMode                   = SortMode::Cpu;
	vk::raii::DescriptorSetLayout        computeDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout             computePipelineLayout      = nullptr;
	vk::raii::Pipeline                   sortInitPipeline           = nullptr;
	vk::raii::Pipeline                   sortBitonicPipeline        = nullptr;
	glm::vec4                            viewRow2{0.0f};        // this frame's view 3rd row, pushed to the sort
	// computeDescriptorSets is declared after descriptorPool below so it destructs first.

	// All splats live in one device-local storage buffer, uploaded once.
	vk::raii::Buffer       splatBuffer       = nullptr;
	vk::raii::DeviceMemory splatBufferMemory = nullptr;

	// Per-frame, host-visible, persistently-mapped sorted-index buffers (back-to-front order).
	// Also serve as the bitonic sort's in-place "values" buffer in GPU mode.
	std::vector<vk::raii::Buffer>       indexBuffers;
	std::vector<vk::raii::DeviceMemory> indexBuffersMemory;
	std::vector<void *>                 indexBuffersMapped;
	std::vector<uint32_t>               sortedIndices;        // CPU scratch reused each frame
	std::vector<float>                  splatDepths;          // CPU scratch reused each frame

	// Per-frame device-local key buffers (compute-only scratch for the bitonic sort).
	std::vector<vk::raii::Buffer>       sortKeyBuffers;
	std::vector<vk::raii::DeviceMemory> sortKeyBuffersMemory;

	std::vector<vk::raii::Buffer>       uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void *>                 uniformBuffersMapped;

	vk::raii::DescriptorPool             descriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;
	std::vector<vk::raii::DescriptorSet> computeDescriptorSets;        // freed before descriptorPool (declared after it)

	vk::raii::CommandPool                commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence>     inFlightFences;
	uint32_t                         frameIndex = 0;

	bool framebufferResized = false;

	// Mouse-look / timing state.
	bool   firstMouse   = true;
	double lastMouseX   = 0.0;
	double lastMouseY   = 0.0;
	float  deltaTime    = 0.0f;
	double lastFrameTime = 0.0;

	// Reconcile COLMAP's Y-down training frame with our Y-up world (toggle with F).
	// 180 deg about X: negates Y and Z. Pure rotation, so covariances stay correct.
	bool sceneUpFlip = true;
	static constexpr glm::mat4 kSceneUpCorrection{1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1};

	glm::mat4 currentViewMatrix() const
	{
		return sceneUpFlip ? camera.viewMatrix() * kSceneUpCorrection : camera.viewMatrix();
	}

	std::vector<const char *> requiredDeviceExtension = {
	    vk::KHRSwapchainExtensionName};

	void loadScene()
	{
		scene      = loadPly(plyPath);
		splatCount = static_cast<uint32_t>(scene.splats.size());
		if (splatCount == 0)
			throw std::runtime_error("ply contained no splats: " + plyPath);
		std::cout << "loaded " << splatCount << " splats from " << plyPath << std::endl;

		// Bitonic sort needs a power-of-two element count; padding slots sort to the end.
		paddedCount = std::max<uint32_t>(2u, std::bit_ceil(splatCount));

		sortedIndices.resize(splatCount);
		splatDepths.resize(splatCount);

		// Place the camera just outside the scene, looking at its center.
		glm::vec3 center = scene.center();
		float     radius = std::max(scene.radius(), 0.5f);
		camera.position  = center + glm::vec3(0.0f, 0.0f, radius * 2.5f);
		camera.yaw       = -glm::half_pi<float>();
		camera.pitch     = 0.0f;
		camera.moveSpeed = radius;        // scale fly speed to the scene
	}

	void initWindow()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Gaussian Splatting Viewer", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

		// Capture the cursor for FPS-style mouse-look.
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwSetCursorPosCallback(window, cursorPosCallback);
		glfwSetKeyCallback(window, keyCallback);
	}

	static void framebufferResizeCallback(GLFWwindow *window, int width, int height)
	{
		auto app                = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	static void cursorPosCallback(GLFWwindow *window, double xpos, double ypos)
	{
		auto app = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
		if (app->firstMouse)
		{
			app->lastMouseX = xpos;
			app->lastMouseY = ypos;
			app->firstMouse = false;
			return;
		}
		float dx        = static_cast<float>(xpos - app->lastMouseX);
		float dy        = static_cast<float>(ypos - app->lastMouseY);
		app->lastMouseX = xpos;
		app->lastMouseY = ypos;
		app->camera.processMouse(dx, dy);
	}

	static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
	{
		auto app = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
		if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GLFW_TRUE);
		if (key == GLFW_KEY_F && action == GLFW_PRESS)
			app->sceneUpFlip = !app->sceneUpFlip;
		if (key == GLFW_KEY_G && action == GLFW_PRESS)
		{
			app->sortMode = (app->sortMode == SortMode::Cpu) ? SortMode::Gpu : SortMode::Cpu;
			std::cout << "sort: " << (app->sortMode == SortMode::Cpu ? "CPU (std::sort)" : "GPU (bitonic)") << std::endl;
		}
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDescriptorSetLayout();
		createComputeDescriptorSetLayout();
		createGraphicsPipeline();
		createComputePipelines();
		createCommandPool();
		createSplatBuffer();
		createIndexBuffers();
		createSortKeyBuffers();
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createComputeDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}

	void mainLoop()
	{
		lastFrameTime = glfwGetTime();
		while (!glfwWindowShouldClose(window))
		{
			double now    = glfwGetTime();
			deltaTime     = static_cast<float>(now - lastFrameTime);
			lastFrameTime = now;

			glfwPollEvents();
			processInput();
			drawFrame();
		}

		device.waitIdle();
	}

	void processInput()
	{
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
			camera.processKeyboard(Camera::Move::Forward, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
			camera.processKeyboard(Camera::Move::Backward, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			camera.processKeyboard(Camera::Move::Left, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			camera.processKeyboard(Camera::Move::Right, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
			camera.processKeyboard(Camera::Move::Up, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
			camera.processKeyboard(Camera::Move::Down, deltaTime);
	}

	void cleanupSwapChain()
	{
		swapChainImageViews.clear();
		swapChain = nullptr;
	}

	void cleanup()
	{
		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		device.waitIdle();

		cleanupSwapChain();
		createSwapChain();
		createImageViews();
	}

	void createInstance()
	{
		constexpr vk::ApplicationInfo appInfo{.pApplicationName   = "Hello Triangle",
		                                      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
		                                      .pEngineName        = "No Engine",
		                                      .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
		                                      .apiVersion         = vk::ApiVersion14};

		std::vector<char const *> requiredLayers;
		if (enableValidationLayers)
		{
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		auto layerProperties    = context.enumerateInstanceLayerProperties();
		auto unsupportedLayerIt = std::ranges::find_if(requiredLayers,
		                                               [&layerProperties](auto const &requiredLayer) {
			                                               return std::ranges::none_of(layerProperties,
			                                                                           [requiredLayer](auto const &layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
		                                               });
		if (unsupportedLayerIt != requiredLayers.end())
		{
			throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIt));
		}

		auto requiredExtensions = getRequiredInstanceExtensions();

		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		auto unsupportedPropertyIt =
		    std::ranges::find_if(requiredExtensions,
		                         [&extensionProperties](auto const &requiredExtension) {
			                         return std::ranges::none_of(extensionProperties,
			                                                     [requiredExtension](auto const &extensionProperty) { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; });
		                         });
		if (unsupportedPropertyIt != requiredExtensions.end())
		{
			throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIt));
		}

		vk::InstanceCreateFlags instanceFlags{};
#ifdef __APPLE__
		// Required alongside VK_KHR_portability_enumeration so the loader enumerates MoltenVK.
		instanceFlags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
#endif

		vk::InstanceCreateInfo createInfo{.flags                   = instanceFlags,
		                                  .pApplicationInfo        = &appInfo,
		                                  .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
		                                  .ppEnabledLayerNames     = requiredLayers.data(),
		                                  .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
		                                  .ppEnabledExtensionNames = requiredExtensions.data()};
		instance = vk::raii::Instance(context, createInfo);
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;

		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
		                                                    vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
		vk::DebugUtilsMessageTypeFlagsEXT     messageTypeFlags(
		    vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
		vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{.messageSeverity = severityFlags,
		                                                                      .messageType     = messageTypeFlags,
		                                                                      .pfnUserCallback = &debugCallback};
		debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
	}

	void createSurface()
	{
		VkSurfaceKHR _surface;
		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		{
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice)
	{
		bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;

		// The GPU sort dispatches on the render queue, so it must also support compute.
		auto queueFamilies    = physicalDevice.getQueueFamilyProperties();
		bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const &qfp) {
			return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) && (qfp.queueFlags & vk::QueueFlagBits::eCompute);
		});

		auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
		bool supportsAllRequiredExtensions =
		    std::ranges::all_of(requiredDeviceExtension,
		                        [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
			                        return std::ranges::any_of(availableDeviceExtensions,
			                                                   [requiredDeviceExtension](auto const &availableDeviceExtension) { return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
		                        });

		auto features                 = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2,
		                                                                     vk::PhysicalDeviceVulkan11Features,
		                                                                     vk::PhysicalDeviceVulkan13Features,
		                                                                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
		bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
		                                features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
		                                features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
		                                features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

		return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
	}

	void pickPhysicalDevice()
	{
		std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
		auto const                            devIter         = std::ranges::find_if(physicalDevices, [&](auto const &physicalDevice) { return isDeviceSuitable(physicalDevice); });
		if (devIter == physicalDevices.end())
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
		physicalDevice = *devIter;
	}

	void createLogicalDevice()
	{
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		// One queue family for graphics, compute and present.
		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
		{
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
			    (queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eCompute) &&
			    physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
			{
				queueIndex = qfpIndex;
				break;
			}
		}
		if (queueIndex == ~0)
		{
			throw std::runtime_error("Could not find a queue for graphics, compute and present -> terminating");
		}

		vk::StructureChain<vk::PhysicalDeviceFeatures2,
		                   vk::PhysicalDeviceVulkan11Features,
		                   vk::PhysicalDeviceVulkan13Features,
		                   vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
		    featureChain = {
		        {},                                                          // vk::PhysicalDeviceFeatures2
		        {.shaderDrawParameters = true},                              // vk::PhysicalDeviceVulkan11Features
		        {.synchronization2 = true, .dynamicRendering = true},        // vk::PhysicalDeviceVulkan13Features
		        {.extendedDynamicState = true}                               // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
		    };

		// A device exposing VK_KHR_portability_subset (MoltenVK) must enable it. The name is a
		// beta-gated constant in vulkan.hpp, so use the literal.
		constexpr char const     *portabilitySubsetExtension = "VK_KHR_portability_subset";
		std::vector<const char *> enabledDeviceExtensions    = requiredDeviceExtension;
		auto                      availableDeviceExtensions  = physicalDevice.enumerateDeviceExtensionProperties();
		if (std::ranges::any_of(availableDeviceExtensions, [&](auto const &ext) { return strcmp(ext.extensionName, portabilitySubsetExtension) == 0; }))
		{
			enabledDeviceExtensions.push_back(portabilitySubsetExtension);
		}

		float                     queuePriority = 0.5f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = queueIndex, .queueCount = 1, .pQueuePriorities = &queuePriority};
		vk::DeviceCreateInfo      deviceCreateInfo{.pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		                                           .queueCreateInfoCount    = 1,
		                                           .pQueueCreateInfos       = &deviceQueueCreateInfo,
		                                           .enabledExtensionCount   = static_cast<uint32_t>(enabledDeviceExtensions.size()),
		                                           .ppEnabledExtensionNames = enabledDeviceExtensions.data()};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);
		queue  = vk::raii::Queue(device, queueIndex, 0);
	}

	void createSwapChain()
	{
		vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
		swapChainExtent                                = chooseSwapExtent(surfaceCapabilities);
		uint32_t minImageCount                         = chooseSwapMinImageCount(surfaceCapabilities);

		std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(*surface);
		swapChainSurfaceFormat                             = chooseSwapSurfaceFormat(availableFormats);

		std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(*surface);
		vk::PresentModeKHR              presentMode           = chooseSwapPresentMode(availablePresentModes);

		vk::SwapchainCreateInfoKHR swapChainCreateInfo{.surface          = *surface,
		                                               .minImageCount    = minImageCount,
		                                               .imageFormat      = swapChainSurfaceFormat.format,
		                                               .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
		                                               .imageExtent      = swapChainExtent,
		                                               .imageArrayLayers = 1,
		                                               .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
		                                               .imageSharingMode = vk::SharingMode::eExclusive,
		                                               .preTransform     = surfaceCapabilities.currentTransform,
		                                               .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
		                                               .presentMode      = presentMode,
		                                               .clipped          = true};

		swapChain       = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	void createImageViews()
	{
		assert(swapChainImageViews.empty());

		vk::ImageViewCreateInfo imageViewCreateInfo{.viewType         = vk::ImageViewType::e2D,
		                                            .format           = swapChainSurfaceFormat.format,
		                                            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
		for (auto &image : swapChainImages)
		{
			imageViewCreateInfo.image = image;
			swapChainImageViews.emplace_back(device, imageViewCreateInfo);
		}
	}

	void createDescriptorSetLayout()
	{
		std::array bindings{
		    // binding 0: camera UBO
		    vk::DescriptorSetLayoutBinding{
		        .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex},
		    // binding 1: splat data (read-only storage buffer)
		    vk::DescriptorSetLayoutBinding{
		        .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex},
		    // binding 2: back-to-front sorted indices (read-only storage buffer)
		    vk::DescriptorSetLayoutBinding{
		        .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex}};
		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	}

	void createComputeDescriptorSetLayout()
	{
		std::array bindings{
		    // binding 0: splat data (read) — positions for depth keys
		    vk::DescriptorSetLayoutBinding{
		        .binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    // binding 1: sort keys (read/write scratch)
		    vk::DescriptorSetLayoutBinding{
		        .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute},
		    // binding 2: sort values == draw indices (read/write, sorted in place)
		    vk::DescriptorSetLayoutBinding{
		        .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eCompute}};
		vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
		computeDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	}

	void createGraphicsPipeline()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/slang.spv"));

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
		vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		// No vertex buffer: quad corners come from gl_VertexIndex, splat data from an SSBO.
		vk::PipelineVertexInputStateCreateInfo   vertexInputInfo{};
		// primitiveRestartEnable must be true for strips on MoltenVK/Metal; harmless for our
		// non-indexed draw.
		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleStrip, .primitiveRestartEnable = vk::True};
		vk::PipelineViewportStateCreateInfo      viewportState{.viewportCount = 1, .scissorCount = 1};

		vk::PipelineRasterizationStateCreateInfo rasterizer{.depthClampEnable        = vk::False,
		                                                    .rasterizerDiscardEnable = vk::False,
		                                                    .polygonMode             = vk::PolygonMode::eFill,
		                                                    .cullMode                = vk::CullModeFlagBits::eNone,        // billboards face the camera
		                                                    .frontFace               = vk::FrontFace::eCounterClockwise,
		                                                    .depthBiasEnable         = vk::False,
		                                                    .lineWidth               = 1.0f};

		vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

		// Premultiplied-alpha "over" compositing for back-to-front splat blending.
		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		    .blendEnable         = vk::True,
		    .srcColorBlendFactor = vk::BlendFactor::eOne,
		    .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		    .colorBlendOp        = vk::BlendOp::eAdd,
		    .srcAlphaBlendFactor = vk::BlendFactor::eOne,
		    .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		    .alphaBlendOp        = vk::BlendOp::eAdd,
		    .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

		vk::PipelineColorBlendStateCreateInfo colorBlending{
		    .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

		std::vector<vk::DynamicState>      dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data()};

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0};
		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
		    {.stageCount          = 2,
		     .pStages             = shaderStages,
		     .pVertexInputState   = &vertexInputInfo,
		     .pInputAssemblyState = &inputAssembly,
		     .pViewportState      = &viewportState,
		     .pRasterizationState = &rasterizer,
		     .pMultisampleState   = &multisampling,
		     .pColorBlendState    = &colorBlending,
		     .pDynamicState       = &dynamicState,
		     .layout              = pipelineLayout,
		     .renderPass          = nullptr},
		    {.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format}};

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	void createComputePipelines()
	{
		vk::raii::ShaderModule shaderModule = createShaderModule(readFile("shaders/compute.spv"));

		vk::PushConstantRange pushRange{.stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = sizeof(SortPushConstants)};
		vk::PipelineLayoutCreateInfo layoutInfo{
		    .setLayoutCount = 1, .pSetLayouts = &*computeDescriptorSetLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange};
		computePipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);

		auto makePipeline = [&](const char *entry) {
			vk::PipelineShaderStageCreateInfo stage{.stage = vk::ShaderStageFlagBits::eCompute, .module = shaderModule, .pName = entry};
			vk::ComputePipelineCreateInfo     info{.stage = stage, .layout = computePipelineLayout};
			return vk::raii::Pipeline(device, nullptr, info);
		};
		sortInitPipeline    = makePipeline("sortInitMain");
		sortBitonicPipeline = makePipeline("sortBitonicMain");
	}

	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		                                   .queueFamilyIndex = queueIndex};
		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
	{
		vk::BufferCreateInfo   bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
		vk::raii::Buffer       buffer          = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size, .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
		vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
		buffer.bindMemory(*bufferMemory, 0);
		return {std::move(buffer), std::move(bufferMemory)};
	}

	void createSplatBuffer()
	{
		vk::DeviceSize bufferSize = sizeof(GpuSplat) * scene.splats.size();

		auto [stagingBuffer, stagingBufferMemory] =
		    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

		void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, scene.splats.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		std::tie(splatBuffer, splatBufferMemory) =
		    createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal);

		copyBuffer(stagingBuffer, splatBuffer, bufferSize);
	}

	void createIndexBuffers()
	{
		// Host-visible, persistently-mapped, one per frame. The CPU sort memcpy's order here; the
		// GPU sort writes it in place as its "values" buffer. Sized to paddedCount for padding.
		vk::DeviceSize bufferSize = sizeof(uint32_t) * paddedCount;
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			auto [buffer, bufferMem] = createBuffer(
			    bufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
			indexBuffers.emplace_back(std::move(buffer));
			indexBuffersMemory.emplace_back(std::move(bufferMem));
			indexBuffersMapped.emplace_back(indexBuffersMemory.back().mapMemory(0, bufferSize));
		}
	}

	void createSortKeyBuffers()
	{
		// Per-frame device-local scratch holding the bitonic sort's depth keys.
		vk::DeviceSize bufferSize = sizeof(uint32_t) * paddedCount;
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			auto [buffer, bufferMem] =
			    createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal);
			sortKeyBuffers.emplace_back(std::move(buffer));
			sortKeyBuffersMemory.emplace_back(std::move(bufferMem));
		}
	}

	void createUniformBuffers()
	{
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
			auto [buffer, bufferMem]  = createBuffer(
			    bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
			uniformBuffers.emplace_back(std::move(buffer));
			uniformBuffersMemory.emplace_back(std::move(bufferMem));
			uniformBuffersMapped.emplace_back(uniformBuffersMemory.back().mapMemory(0, bufferSize));
		}
	}

	void createDescriptorPool()
	{
		std::array poolSizes{
		    vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = MAX_FRAMES_IN_FLIGHT},
		    // graphics: splat + index per frame (2); compute: splat + keys + values per frame (3)
		    vk::DescriptorPoolSize{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 5 * MAX_FRAMES_IN_FLIGHT}};
		vk::DescriptorPoolCreateInfo poolInfo{.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		                                      .maxSets       = 2 * MAX_FRAMES_IN_FLIGHT,        // graphics set + compute set per frame
		                                      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		                                      .pPoolSizes    = poolSizes.data()};
		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
	}

	void createDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool     = descriptorPool,
		                                               .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		                                               .pSetLayouts        = layouts.data()};

		descriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DescriptorBufferInfo uboInfo{.buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject)};
			vk::DescriptorBufferInfo splatInfo{.buffer = splatBuffer, .offset = 0, .range = sizeof(GpuSplat) * splatCount};
			vk::DescriptorBufferInfo indexInfo{.buffer = indexBuffers[i], .offset = 0, .range = sizeof(uint32_t) * splatCount};

			std::array descriptorWrites{
			    vk::WriteDescriptorSet{.dstSet = descriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &uboInfo},
			    vk::WriteDescriptorSet{.dstSet = descriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &splatInfo},
			    vk::WriteDescriptorSet{.dstSet = descriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &indexInfo}};
			device.updateDescriptorSets(descriptorWrites, {});
		}
	}

	void createComputeDescriptorSets()
	{
		std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *computeDescriptorSetLayout);
		vk::DescriptorSetAllocateInfo        allocInfo{.descriptorPool     = descriptorPool,
		                                               .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		                                               .pSetLayouts        = layouts.data()};

		computeDescriptorSets = device.allocateDescriptorSets(allocInfo);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vk::DescriptorBufferInfo splatInfo{.buffer = splatBuffer, .offset = 0, .range = sizeof(GpuSplat) * splatCount};
			vk::DescriptorBufferInfo keyInfo{.buffer = sortKeyBuffers[i], .offset = 0, .range = sizeof(uint32_t) * paddedCount};
			vk::DescriptorBufferInfo valueInfo{.buffer = indexBuffers[i], .offset = 0, .range = sizeof(uint32_t) * paddedCount};

			std::array descriptorWrites{
			    vk::WriteDescriptorSet{.dstSet = computeDescriptorSets[i], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &splatInfo},
			    vk::WriteDescriptorSet{.dstSet = computeDescriptorSets[i], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &keyInfo},
			    vk::WriteDescriptorSet{.dstSet = computeDescriptorSets[i], .dstBinding = 2, .dstArrayElement = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &valueInfo}};
			device.updateDescriptorSets(descriptorWrites, {});
		}
	}

	void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer, vk::DeviceSize size)
	{
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
		vk::raii::CommandBuffer       commandCopyBuffer = std::move(device.allocateCommandBuffers(allocInfo).front());
		commandCopyBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
		commandCopyBuffer.end();
		queue.submit(vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
		queue.waitIdle();
	}

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createCommandBuffers()
	{
		commandBuffers.clear();
		vk::CommandBufferAllocateInfo allocInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT};
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

	// Record the GPU bitonic sort, leaving back-to-front draw indices in indexBuffers[frameIndex].
	void recordSort()
	{
		auto &commandBuffer = commandBuffers[frameIndex];

		SortPushConstants pc{.viewRow2 = viewRow2, .splatCount = splatCount, .paddedCount = paddedCount, .j = 0, .k = 0};

		// Serializes dependent dispatches (write -> read/write).
		const vk::MemoryBarrier2 computeBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite};
		const vk::DependencyInfo computeDependency{.memoryBarrierCount = 1, .pMemoryBarriers = &computeBarrier};

		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, computePipelineLayout, 0, *computeDescriptorSets[frameIndex], nullptr);

		// 1) Fill keys (order-preserving depth) and identity values for every padded slot.
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *sortInitPipeline);
		commandBuffer.pushConstants<SortPushConstants>(computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);
		uint32_t groups = (paddedCount + 255u) / 256u;
		commandBuffer.dispatch(groups, 1, 1);
		commandBuffer.pipelineBarrier2(computeDependency);

		// 2) Bitonic network: one dispatch per (k, j) stage, in place on keys + values.
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *sortBitonicPipeline);
		for (uint32_t k = 2; k <= paddedCount; k <<= 1)
		{
			for (uint32_t j = k >> 1; j > 0; j >>= 1)
			{
				pc.k = k;
				pc.j = j;
				commandBuffer.pushConstants<SortPushConstants>(computePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);
				commandBuffer.dispatch(groups, 1, 1);
				commandBuffer.pipelineBarrier2(computeDependency);
			}
		}

		// 3) Hand the sorted indices to the vertex stage.
		const vk::MemoryBarrier2 toVertexBarrier{
		    .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
		    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
		    .dstStageMask  = vk::PipelineStageFlagBits2::eVertexShader,
		    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead};
		commandBuffer.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &toVertexBarrier});
	}

	void recordCommandBuffer(uint32_t imageIndex)
	{
		auto &commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});

		if (sortMode == SortMode::Gpu)
			recordSort();

		transition_image_layout(
		    imageIndex,
		    vk::ImageLayout::eUndefined,
		    vk::ImageLayout::eColorAttachmentOptimal,
		    {},                                                        // srcAccessMask (no need to wait for previous operations)
		    vk::AccessFlagBits2::eColorAttachmentWrite,                // dstAccessMask
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput         // dstStage
		);
		vk::ClearValue              clearColor     = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::RenderingAttachmentInfo attachmentInfo = {
		    .imageView   = swapChainImageViews[imageIndex],
		    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		    .loadOp      = vk::AttachmentLoadOp::eClear,
		    .storeOp     = vk::AttachmentStoreOp::eStore,
		    .clearValue  = clearColor};
		vk::RenderingInfo renderingInfo = {
		    .renderArea           = {.offset = {0, 0}, .extent = swapChainExtent},
		    .layerCount           = 1,
		    .colorAttachmentCount = 1,
		    .pColorAttachments    = &attachmentInfo};
		commandBuffer.beginRendering(renderingInfo);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
		commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[frameIndex], nullptr);
		// One instanced quad (4-vert triangle strip) per splat.
		commandBuffer.draw(4, splatCount, 0, 0);
		commandBuffer.endRendering();

		transition_image_layout(
		    imageIndex,
		    vk::ImageLayout::eColorAttachmentOptimal,
		    vk::ImageLayout::ePresentSrcKHR,
		    vk::AccessFlagBits2::eColorAttachmentWrite,                // srcAccessMask
		    {},                                                        // dstAccessMask
		    vk::PipelineStageFlagBits2::eColorAttachmentOutput,        // srcStage
		    vk::PipelineStageFlagBits2::eBottomOfPipe                  // dstStage
		);
		commandBuffer.end();
	}

	void transition_image_layout(
	    uint32_t                imageIndex,
	    vk::ImageLayout         old_layout,
	    vk::ImageLayout         new_layout,
	    vk::AccessFlags2        src_access_mask,
	    vk::AccessFlags2        dst_access_mask,
	    vk::PipelineStageFlags2 src_stage_mask,
	    vk::PipelineStageFlags2 dst_stage_mask)
	{
		vk::ImageMemoryBarrier2 barrier = {
		    .srcStageMask        = src_stage_mask,
		    .srcAccessMask       = src_access_mask,
		    .dstStageMask        = dst_stage_mask,
		    .dstAccessMask       = dst_access_mask,
		    .oldLayout           = old_layout,
		    .newLayout           = new_layout,
		    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		    .image               = swapChainImages[imageIndex],
		    .subresourceRange    = {
		        .aspectMask     = vk::ImageAspectFlagBits::eColor,
		        .baseMipLevel   = 0,
		        .levelCount     = 1,
		        .baseArrayLayer = 0,
		        .layerCount     = 1}};
		vk::DependencyInfo dependency_info = {
		    .dependencyFlags         = {},
		    .imageMemoryBarrierCount = 1,
		    .pImageMemoryBarriers    = &barrier};
		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
	}

	void createSyncObjects()
	{
		assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
			inFlightFences.emplace_back(device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
		}
	}

	static constexpr float kFovY = glm::radians(60.0f);

	void updateUniformBuffer(uint32_t currentImage)
	{
		float width  = static_cast<float>(swapChainExtent.width);
		float height = static_cast<float>(swapChainExtent.height);

		UniformBufferObject ubo{};
		ubo.view = currentViewMatrix();
		ubo.proj = glm::perspective(kFovY, width / height, 0.05f, 1000.0f);
		ubo.proj[1][1] *= -1;        // GLM assumes OpenGL's flipped Y

		// Focal lengths in pixels, consistent with the covariance projection Jacobian.
		float fy       = height / (2.0f * std::tan(kFovY * 0.5f));
		ubo.viewport   = glm::vec2(width, height);
		ubo.focal      = glm::vec2(fy, fy);

		memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
	}

	// The view matrix 3rd row: camera-space z = dot(row2.xyz, pos) + row2.w. With -Z forward,
	// more-negative z is farther away, so ascending z is back-to-front order for "over" blending.
	// Shared by both sort paths (CPU below, GPU push constants) so their keys match exactly.
	void updateViewRow2()
	{
		glm::mat4 view = currentViewMatrix();
		viewRow2       = glm::vec4(view[0][2], view[1][2], view[2][2], view[3][2]);
	}

	// CPU path: sort splats back-to-front by camera-space depth and upload the order.
	void sortSplats()
	{
		for (uint32_t i = 0; i < splatCount; ++i)
		{
			const glm::vec3 &p = scene.splats[i].position;
			splatDepths[i]     = viewRow2.x * p.x + viewRow2.y * p.y + viewRow2.z * p.z + viewRow2.w;
			sortedIndices[i]   = i;
		}
		std::ranges::sort(sortedIndices, [this](uint32_t a, uint32_t b) { return splatDepths[a] < splatDepths[b]; });

		memcpy(indexBuffersMapped[frameIndex], sortedIndices.data(), sizeof(uint32_t) * splatCount);
	}

	void drawFrame()
	{
		// Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are indexed by frameIndex,
		//       while renderFinishedSemaphores is indexed by imageIndex
		auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (fenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to wait for fence!");
		}

		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

		// VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS lets eErrorOutOfDateKHR return here
		// instead of throwing.
		if (result == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapChain();
			return;
		}
		if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
		{
			assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
			throw std::runtime_error("failed to acquire swap chain image!");
		}
		updateUniformBuffer(frameIndex);
		updateViewRow2();
		if (sortMode == SortMode::Cpu)
			sortSplats();
		// GPU mode sorts inside the command buffer (recordSort), using viewRow2 as push constants.

		// Only reset the fence if we are submitting work
		device.resetFences(*inFlightFences[frameIndex]);

		commandBuffers[frameIndex].reset();
		recordCommandBuffer(imageIndex);

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		const vk::SubmitInfo   submitInfo{.waitSemaphoreCount   = 1,
		                                  .pWaitSemaphores      = &*presentCompleteSemaphores[frameIndex],
		                                  .pWaitDstStageMask    = &waitDestinationStageMask,
		                                  .commandBufferCount   = 1,
		                                  .pCommandBuffers      = &*commandBuffers[frameIndex],
		                                  .signalSemaphoreCount = 1,
		                                  .pSignalSemaphores    = &*renderFinishedSemaphores[imageIndex]};
		queue.submit(submitInfo, *inFlightFences[frameIndex]);

		const vk::PresentInfoKHR presentInfoKHR{.waitSemaphoreCount = 1,
		                                        .pWaitSemaphores    = &*renderFinishedSemaphores[imageIndex],
		                                        .swapchainCount     = 1,
		                                        .pSwapchains        = &*swapChain,
		                                        .pImageIndices      = &imageIndex};
		result = queue.presentKHR(presentInfoKHR);
		if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else
		{
			assert(result == vk::Result::eSuccess);
		}
		frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
	{
		vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size(), .pCode = reinterpret_cast<const uint32_t *>(code.data())};
		vk::raii::ShaderModule     shaderModule{device, createInfo};

		return shaderModule;
	}

	static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const &surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
		{
			minImageCount = surfaceCapabilities.maxImageCount;
		}
		return minImageCount;
	}

	static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
	{
		assert(!availableFormats.empty());
		const auto formatIt = std::ranges::find_if(
		    availableFormats,
		    [](const auto &format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear; });
		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}

	static vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const &availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
		return std::ranges::any_of(availablePresentModes,
		                           [](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
		           vk::PresentModeKHR::eMailbox :
		           vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {
		    std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		    std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
	}

	std::vector<const char *> getRequiredInstanceExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		auto     glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		if (enableValidationLayers)
		{
			extensions.push_back(vk::EXTDebugUtilsExtensionName);
		}
#ifdef __APPLE__
		// MoltenVK is a non-conformant (portability) driver; the loader hides it unless
		// the application opts in via the portability enumeration extension + flag.
		extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);
		extensions.push_back(vk::KHRGetPhysicalDeviceProperties2ExtensionName);
#endif

		return extensions;
	}

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *)
	{
		if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
		{
			std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
		}

		return vk::False;
	}

	static std::vector<char> readFile(const std::string &filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}
		std::vector<char> buffer(file.tellg());
		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
		file.close();
		return buffer;
	}
};

int main(int argc, char **argv)
{
	// Path to a pretrained 3DGS .ply; defaults to the bundled cactus scene.
	std::string plyPath = (argc > 1) ? argv[1] : "models/cactus.ply";

	try
	{
		Application app(plyPath);
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
