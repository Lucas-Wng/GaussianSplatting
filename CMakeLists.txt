cmake_minimum_required (VERSION 3.29)

set (TEMPLATE_NAME "GaussianSplatting")

project (${TEMPLATE_NAME})

# Add option to enable/disable C++ 20 module
option(ENABLE_CPP20_MODULE "Enable C++ 20 module support for Vulkan" OFF)

# Enable C++ module dependency scanning only if C++ 20 module is enabled
if(ENABLE_CPP20_MODULE)
  set(CMAKE_CXX_SCAN_FOR_MODULES ON)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan 1.4.335 REQUIRED)

# Set up Vulkan C++ module only if enabled
if(ENABLE_CPP20_MODULE)
  add_library(VulkanCppModule)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)

  target_compile_definitions(VulkanCppModule PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1)
  target_include_directories(VulkanCppModule PUBLIC "${Vulkan_INCLUDE_DIR}")
  target_link_libraries(VulkanCppModule PUBLIC Vulkan::Vulkan)

  set_target_properties(VulkanCppModule PROPERTIES CXX_STANDARD 20)

  # Add MSVC-specific compiler options for proper C++ module support
  if(MSVC)
    target_compile_options(VulkanCppModule PRIVATE
      /std:c++latest      # Use latest C++ standard for better module support
      /permissive-        # Standards conformance mode
      /Zc:__cplusplus     # Enable correct __cplusplus macro
      /EHsc               # Enable C++ exception handling
      /Zc:preprocessor    # Use conforming preprocessor
      /translateInclude   # Automatically translate #include to import for standard library
    )
  endif()

  target_sources(VulkanCppModule
    PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES
    BASE_DIRS
    "${Vulkan_INCLUDE_DIR}"
    FILES
    "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
  )


  # Add the vulkan.cppm file directly as a source file
  target_sources(VulkanCppModule
    PRIVATE
    "${Vulkan_INCLUDE_DIR}/vulkan/vulkan.cppm"
  )
else()
  # Create a dummy interface library when C++ 20 module is disabled
  add_library(VulkanCppModule INTERFACE)
  add_library(Vulkan::cppm ALIAS VulkanCppModule)
  target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
  target_compile_definitions(VulkanCppModule
    INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
  )
endif()

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

set(TEMPLATE_DIR ${CMAKE_BINARY_DIR}/${TEMPLATE_NAME})

add_executable (${TEMPLATE_NAME} src/main.cpp src/ply_loader.cpp)
set_target_properties (${TEMPLATE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEMPLATE_DIR})
set_target_properties (${TEMPLATE_NAME} PROPERTIES CXX_STANDARD 20)
target_link_libraries (${TEMPLATE_NAME} Vulkan::cppm glfw glm::glm)

# Add compile definition if C++ 20 module is enabled
if(ENABLE_CPP20_MODULE)
  target_compile_definitions(${TEMPLATE_NAME} PRIVATE USE_CPP20_MODULES=1)
endif()

# Define VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS to treat VK_ERROR_OUT_OF_DATE_KHR as a success code
target_compile_definitions(${TEMPLATE_NAME} PRIVATE "VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS" )

if(WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set_target_properties(${TEMPLATE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY ${TEMPLATE_DIR})
  endif()
endif()

# Slang shaders
set (SHADER_TARGET ${TEMPLATE_NAME}Shader)
set (SHADERS_DIR ${TEMPLATE_DIR}/shaders)
add_custom_command(OUTPUT ${SHADERS_DIR} COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR})

# Graphics pipeline: EWA splat vertex + fragment shaders.
set (SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang")
set (ENTRY_POINTS -entry vertMain -entry fragMain)
add_custom_command(
  OUTPUT  ${SHADERS_DIR}/slang.spv
  COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o ${TEMPLATE_DIR}/shaders/slang.spv
  WORKING_DIRECTORY ${SHADERS_DIR}
  DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
  COMMENT "Compiling Slang Shaders"
  VERBATIM
)

# Compute pipeline: GPU bitonic depth sort.
set (COMPUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.slang")
set (COMPUTE_ENTRY_POINTS -entry sortInitMain -entry sortBitonicMain)
add_custom_command(
  OUTPUT  ${SHADERS_DIR}/compute.spv
  COMMAND ${SLANGC_EXECUTABLE} ${COMPUTE_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${COMPUTE_ENTRY_POINTS} -o ${TEMPLATE_DIR}/shaders/compute.spv
  WORKING_DIRECTORY ${SHADERS_DIR}
  DEPENDS ${SHADERS_DIR} ${COMPUTE_SOURCES}
  COMMENT "Compiling Slang Compute Shaders"
  VERBATIM
)

# Compute pipeline: LSD radix sort + exclusive scan.
set (RADIX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/radix.slang")
set (RADIX_ENTRY_POINTS -entry radixHistogramMain -entry radixScatterMain -entry scanBlockMain -entry scanAddMain -entry radixDepthInitMain)
add_custom_command(
  OUTPUT  ${SHADERS_DIR}/radix.spv
  COMMAND ${SLANGC_EXECUTABLE} ${RADIX_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${RADIX_ENTRY_POINTS} -o ${TEMPLATE_DIR}/shaders/radix.spv
  WORKING_DIRECTORY ${SHADERS_DIR}
  DEPENDS ${SHADERS_DIR} ${RADIX_SOURCES}
  COMMENT "Compiling Slang Radix Shaders"
  VERBATIM
)

add_custom_target(${SHADER_TARGET} DEPENDS ${SHADERS_DIR}/slang.spv ${SHADERS_DIR}/compute.spv ${SHADERS_DIR}/radix.spv)
add_dependencies(${TEMPLATE_NAME} ${SHADER_TARGET})

# Bundle every .ply in assets/ into the run directory's models/ folder
file (GLOB SCENE_PLYS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/assets/*.ply")
if (SCENE_PLYS)
  file (COPY ${SCENE_PLYS} DESTINATION ${TEMPLATE_DIR}/models)
endif()
