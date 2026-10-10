#pragma once
#include <algorithm>
#include <cstddef>

// Counts nonempty compute epochs separated by explicit compute->compute
// execution dependencies. Unannotated/manual boundaries are ignored, not
// invented: the count is structurally conservative, not a timing assertion.
struct vk_tp5_dependency_ledger {
    size_t dispatches=0, epochs=0, width=0, maximum_width=0;
    void boundary(){if(width){++epochs;maximum_width=std::max(maximum_width,width);width=0;}}
    void note(bool dispatch,bool dependency){if(dependency)boundary();if(dispatch){++dispatches;++width;}}
    void finish(){boundary();}
};
