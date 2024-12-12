#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_core.h>

// PFN_vkDebugUtilsMessengerCallbackEXT callback
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT msg_severity,
              VkDebugUtilsMessageTypeFlagsEXT msg_type,
              const VkDebugUtilsMessengerCallbackDataEXT *p_clbk_data,
              void *p_user_data) {
  if (msg_severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    std::cerr << "validation layer: " << p_clbk_data->pMessage << std::endl;
  }
  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance m_instance,
    const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      m_instance, "vkCreateDebugUtilsMessengerEXT");
  return nullptr != func
             ? func(m_instance, pCreateInfo, pAllocator, pDebugMessenger)
             : VK_ERROR_EXTENSION_NOT_PRESENT;
}

void PopulateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
  createInfo.pUserData = nullptr;
}

void DestroyDebugUtilsMessengerEXT(VkInstance m_instance,
                                   VkDebugUtilsMessengerEXT m_debugMessenger,
                                   const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      m_instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(m_instance, m_debugMessenger, pAllocator);
  }
}

struct QueueFamilyIndices {
  std::optional<uint32_t> graphics_family;
  std::optional<uint32_t> presentation_family;

  bool isComplete() {
    return graphics_family.has_value() && presentation_family.has_value();
  }
};

class HelloTriangleApp {
private:
  const uint32_t WIDTH{800};
  const uint32_t HEIGHT{600};
  const std::vector<const char *> m_validationLayers{
      "VK_LAYER_KHRONOS_validation"};

  GLFWwindow *m_window;
  VkInstance m_instance;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VkDevice m_device;
  VkQueue m_graphicsQueue;
  VkQueue m_presentQueue;
  VkDebugUtilsMessengerEXT m_debugMessenger;
  VkSurfaceKHR m_surface;

public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    m_window =
        glfwCreateWindow(WIDTH, HEIGHT, "Hello Triangle", nullptr, nullptr);
    if (m_window == NULL) {
      throw std::runtime_error("m_window creation failed");
    }
  }

  void initVulkan() {
#ifndef NDEBUG
    // vulkan validation layers
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    for (auto layer_name : m_validationLayers) {
      bool layer_found = false;
      for (const auto &layer_props : availableLayers) {
        if (strcmp(layer_name, layer_props.layerName)) {
          layer_found = true;
          break;
        }
      }
      if (!layer_name) {
        std::stringstream ss;
        ss << "requested validation layer: " << layer_name << " was not found";
        throw std::runtime_error(ss.str());
      }
    }
#endif

    // Vulkan application info to be provided to Vulkan m_instance creation
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    // query supported extensions and list them
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> supported_extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount,
                                           supported_extensions.data());
    std::cout << "available extensions:" << std::endl;
    for (const auto &extension : supported_extensions) {
      std::cout << "\t" << extension.extensionName << std::endl;
    }

    uint32_t glfwExtensionCount{0};
    const char **glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char *> extensions(glfwExtensions,
                                         glfwExtensions + glfwExtensionCount);
    VkInstanceCreateInfo createVkInstanceInfo{};
#ifdef NDEBUG
    createVkInstanceInfo.enabledLayerCount = 0;
#else
    VkDebugUtilsMessengerCreateInfoEXT createDebugMessengerInfo{};
    PopulateDebugMessengerCreateInfo(createDebugMessengerInfo);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    createVkInstanceInfo.ppEnabledLayerNames = m_validationLayers.data();
    createVkInstanceInfo.enabledLayerCount =
        static_cast<uint32_t>(m_validationLayers.size());
#endif
    createVkInstanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createVkInstanceInfo.pApplicationInfo = &appInfo;
    createVkInstanceInfo.enabledExtensionCount =
        static_cast<uint32_t>(extensions.size());
    createVkInstanceInfo.ppEnabledExtensionNames = extensions.data();
    if (vkCreateInstance(&createVkInstanceInfo, nullptr, &m_instance) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create a Vk m_instance!");
    }

#ifndef NDEBUG
    if (CreateDebugUtilsMessengerEXT(m_instance, &createDebugMessengerInfo,
                                     nullptr, &m_debugMessenger) != VK_SUCCESS)
      throw std::runtime_error("failed to set up debug messenger!");
#endif
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
  }

  void createSurface() {
    if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create window surface!");
    }
  }

  void pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (0 == deviceCount) {
      throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    const auto is_device_suitable = [this](VkPhysicalDevice device) {
      // query name, type, supported Vulkan version, etc ...
      VkPhysicalDeviceProperties device_props;
      vkGetPhysicalDeviceProperties(device, &device_props);
      // query support for texture compression, 64bit floats, mutiviewport
      // rendering, etc ...
      VkPhysicalDeviceFeatures device_features;
      vkGetPhysicalDeviceFeatures(device, &device_features);

      // find a suitable queue family
      QueueFamilyIndices indices = this->findQueueFamilies(device);
      return indices.isComplete();
    };

    for (const auto &device : devices) {
      if (is_device_suitable(device)) {
        m_physicalDevice = device;
        break;
      }
    }
    if (m_physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
  }

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    uint32_t queue_family_count{0};
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count,
                                             queueFamilies.data());
    for (size_t i{0}; i < queue_family_count; ++i) {
      if (queueFamilies.at(i).queueFlags & VK_QUEUE_GRAPHICS_BIT) {
        indices.graphics_family = i;
      }
      VkBool32 presentation_support = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface,
                                           &presentation_support);
      if (presentation_support) {
        indices.presentation_family = i;
      }
      if (indices.isComplete())
        break;
    }
    return indices;
  }

  void createLogicalDevice() {
    // specify the queues to be created
    QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
    if (!indices.isComplete()) {
      throw std::runtime_error("no suitable queue family was found!");
    }

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilyIndices{
        indices.graphics_family.value(), indices.presentation_family.value()};
    // TODO: print the queue family indices set

    float queuePriority{1.0f};
    for (uint32_t queueFamilyIdx : uniqueQueueFamilyIndices) {
      VkDeviceQueueCreateInfo queueCreateInfo{};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamilyIdx;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    }

    // specify used device features
    VkPhysicalDeviceFeatures deviceFeatures{};

    // create the logical device - device level validation layers are deprecated
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;
    createInfo.enabledLayerCount = 0;
    if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) !=
        VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device!");
    }

    // retrieve family queues
    vkGetDeviceQueue(m_device, indices.graphics_family.value(), 0,
                     &m_graphicsQueue);
    vkGetDeviceQueue(m_device, indices.presentation_family.value(), 0,
                     &m_presentQueue);
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(m_window)) {
      glfwPollEvents();
    }
  }

  void cleanup() {
    vkDestroyDevice(m_device, nullptr);
#ifndef NDEBUG
    DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
#endif
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    vkDestroyInstance(m_instance, nullptr);
    glfwDestroyWindow(m_window);
    glfwTerminate();
  }
};

int main() {
  HelloTriangleApp app;
  try {
    app.run();
  } catch (const std::exception &e) {
    std::cout << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
