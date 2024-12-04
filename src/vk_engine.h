// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <camera.h>
#include <vk_descriptors.h>
#include <vk_loader.h>
#include <vk_types.h>

struct DeletionQueue {
    std::deque<std::function<void()>> deletors;

    void push_function(std::function<void()> &&function) {
        deletors.push_back(function);
    }

    void flush() {
        // reverse iterate the deletion queue to execute all the functions
        for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
            (*it)();  // call functors
        }

        deletors.clear();
    }
};

struct ComputePushConstants {
    glm::vec4 data1;
    glm::vec4 data2;
    glm::vec4 data3;
    glm::vec4 data4;
};

struct ComputeEffect {
    const char *name;

    VkPipeline pipeline;
    VkPipelineLayout layout;

    ComputePushConstants data;
};

struct FrameData {
    VkSemaphore _swapchainSemaphore, _renderSemaphore;
    VkFence _renderFence;

    VkCommandPool _commandPool;
    VkCommandBuffer _mainCommandBuffer;

    DeletionQueue _deletionQueue;
    DescriptorAllocatorGrowable _frameDescriptors;
};

struct GPUSceneData {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewproj;
    glm::vec4 ambientColor;
    glm::vec4 sunlightDirection;  // w for sun power
    glm::vec4 sunlightColor;
};

struct GLTFMetallic_Roughness {
    MaterialPipeline opaquePipeline;
    MaterialPipeline transparentPipeline;

    VkDescriptorSetLayout materialLayout;

    struct MaterialConstants {
        glm::vec4 colorFactors;
        glm::vec4 metal_rough_factors;
        // padding, we need it anyway for uniform buffers
        glm::vec4 extra[14];
    };

    struct MaterialResources {
        AllocatedImage colorImage;
        VkSampler colorSampler;
        AllocatedImage metalRoughImage;
        VkSampler metalRoughSampler;
        VkBuffer dataBuffer;
        uint32_t dataBufferOffset;
    };

    DescriptorWriter writer;

    void build_pipelines(VulkanEngine *engine);
    void clear_resources(VkDevice device);

    MaterialInstance write_material(VkDevice device, MaterialPass pass, const MaterialResources &resources,
                                    DescriptorAllocatorGrowable &descriptorAllocator);
};

struct MeshNode : public Node {
    std::shared_ptr<MeshAsset> mesh;

    virtual void Draw(const glm::mat4 &topMatrix, DrawContext &ctx) override;
};

struct RenderObject {
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;

    MaterialInstance *material;

    glm::mat4 transform;
    VkDeviceAddress vertexBufferAddress;
};

struct DrawContext {
    std::vector<RenderObject> OpaqueSurfaces;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:
    bool _isInitialized{false};
    int _frameNumber{0};
    bool stop_rendering{false};
    VkExtent2D _windowExtent{1700, 900};

    struct SDL_Window *_window{nullptr};

    static VulkanEngine &Get();

    // initializes everything in the engine
    void init();

    // shuts down the engine
    void cleanup();

    // draw loop
    void draw();

    // draw background
    void draw_background(VkCommandBuffer cmd);

    // draw triangle
    void draw_geometry(VkCommandBuffer cmd);

    // update scene
    void update_scene();

    // run main loop
    void run();

    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);

    AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
    void destroy_buffer(const AllocatedBuffer &buffer);

    // load textures
    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void *data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage,
                                bool mipmapped = false);
    void destroy_image(const AllocatedImage &img);

    VmaAllocator _allocator;

    // draw resources
    AllocatedImage _drawImage;
    AllocatedImage _depthImage;
    VkExtent2D _drawExtent;
    float renderScale = 1.f;

    // global descriptor allocator
    DescriptorAllocatorGrowable globalDescriptorAllocator;

    VkDescriptorSet _drawImageDescriptors;
    VkDescriptorSetLayout _drawImageDescriptorLayout;
    VkDescriptorSetLayout _singleImageDescriptorLayout;

    // compute pipeline
    VkPipeline _gradientPipeline;
    VkPipelineLayout _gradientPipelineLayout;

    // immediate submit structures
    VkFence _immFence;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;

    void immediate_submit(std::function<void(VkCommandBuffer cmd)> &&function);

    std::vector<ComputeEffect> backgroundEffects;
    int currentBackgroundEffect{0};

    // triangle pipeline
    VkPipelineLayout _trianglePipelineLayout;
    VkPipeline _trianglePipeline;

    // mesh pipeline
    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _meshPipeline;

    GPUMeshBuffers rectangle;

    // mesh
    std::vector<std::shared_ptr<MeshAsset>> testMeshes;

    // scene
    GPUSceneData sceneData;
    VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;

    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    VkSampler _defaultSamplerLinear;
    VkSampler _defaultSamplerNearest;

    // material
    MaterialInstance defaultData;
    GLTFMetallic_Roughness metalRoughMaterial;

    // nodes
    DrawContext mainDrawContext;
    std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;

    // camera
    Camera mainCamera;

    VkInstance _instance;                       // Vulkan library handle
    VkDebugUtilsMessengerEXT _debug_messenger;  // Vulkan debug output handle
    VkPhysicalDevice _chosenGPU;                // GPU chosen as the default device
    VkDevice _device;                           // Vulkan device for commands
    VkSurfaceKHR _surface;                      // Vulkan window surface

    VkSwapchainKHR _swapchain;
    VkFormat _swapchainImageFormat;

    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    VkExtent2D _swapchainExtent;

    FrameData _frames[FRAME_OVERLAP];

    FrameData &get_current_frame() {
        return _frames[_frameNumber % FRAME_OVERLAP];
    };

    DeletionQueue _mainDeletionQueue;
    VkQueue _graphicsQueue;
    uint32_t _graphicsQueueFamily;

    bool resize_requested{false};

private:
    void init_vulkan();
    void init_swapchain();
    void resize_swapchain();
    void init_commands();
    void init_sync_structures();

    void create_swapchain(uint32_t width, uint32_t height);
    void destroy_swapchain();

    void init_descriptors();

    void init_pipelines();
    void init_background_pipelines();

    void init_triangle_pipeline();

    void init_mesh_pipeline();
    void init_default_data();

    void init_imgui();
    void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
};
