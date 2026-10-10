#include "ggml-vulkan-tp5-sparse-timing.h"
#include "ggml-vulkan-tp5-barrier-key.h"
#include <cassert>
#include <cstdio>

int main() {
    const auto dense = vk_tp5_sparse_timing::parse(nullptr);
    const auto span = vk_tp5_sparse_timing::parse("span");
    unsigned count = 0;
    for (unsigned q = 0; q < 483; ++q) { assert(dense.query(q,96)); count += span.query(q,96); }
    assert(count == 2 && !span.query(483,96));
    for (const char * bad : {"", "0:0", "8:8", "-1:0", "8:0junk", "8:", "98:0", "1:0:0"})
        assert(!vk_tp5_sparse_timing::parse(bad).valid);
    for (unsigned stage = 0; stage <= 96; ++stage) {
        unsigned cover = 0;
        for (unsigned residue = 0; residue < 8; ++residue)
            cover += vk_tp5_sparse_timing::parse(("8:" + std::to_string(residue)).c_str()).stage(stage);
        assert(cover == 1);
    }
    auto a = vk_tp5_barrier_key{};
    a.src = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT; a.dst = a.src;
    a.memory.push_back({VK_STRUCTURE_TYPE_MEMORY_BARRIER,nullptr,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT});
    a.buffers.push_back({VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,nullptr,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED,VK_QUEUE_FAMILY_IGNORED,(VkBuffer)(uintptr_t)1,32,64});
    assert(a==a);
    auto b=a; b.src=VK_PIPELINE_STAGE_TRANSFER_BIT; assert(!(a==b));
    b=a;b.dst=VK_PIPELINE_STAGE_TRANSFER_BIT;assert(!(a==b));
    b=a;b.flags=VK_DEPENDENCY_BY_REGION_BIT;assert(!(a==b));
    b=a;b.memory[0].srcAccessMask=0;assert(!(a==b));
    b=a;b.memory[0].dstAccessMask=0;assert(!(a==b));
    b=a;b.buffers[0].buffer=(VkBuffer)(uintptr_t)2;assert(!(a==b));
    b=a;b.buffers[0].offset++;assert(!(a==b));
    b=a;b.buffers[0].size++;assert(!(a==b));
    b=a;b.buffers[0].srcAccessMask=0;assert(!(a==b));
    b=a;b.buffers[0].dstAccessMask=0;assert(!(a==b));
    b=a;b.buffers[0].srcQueueFamilyIndex=1;assert(!(a==b));
    b=a;b.buffers[0].dstQueueFamilyIndex=1;assert(!(a==b));
    b=a;b.buffers[0].pNext=&a;assert(!(a==b));
    b=a;b.memory[0].pNext=&a;assert(!(a==b));
    b=a;b.memory.clear();assert(!(a==b));
    std::puts("Sparse queries partition 97 stages; missing queries excluded; barrier identities/scopes/ownership/extensions guarded.");
}
