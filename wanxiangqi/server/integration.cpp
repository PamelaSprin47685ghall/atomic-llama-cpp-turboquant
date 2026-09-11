#include "integration.h"

bool wanxiangqi_server_token_dump::init(const common_params & params, bool is_resume) {
    if (is_resume) {
        return true;
    }

    impl_.reset();
    if (params.request_token_dump_path.empty()) {
        return true;
    }

    try {
        impl_ = std::make_unique<server_token_dump>(
            params.request_token_dump_path,
            params.request_token_dump_shard_records);
        SRV_INF("request token dump enabled: path=%s, nodes=%llu, requests=%llu\n",
                params.request_token_dump_path.c_str(),
                (unsigned long long) impl_->node_count(),
                (unsigned long long) impl_->request_count());
        return true;
    } catch (const std::exception & e) {
        SRV_ERR("failed to initialize request token dump: %s\n", e.what());
        return false;
    }
}

void wanxiangqi_server_token_dump::append(
        const std::vector<llama_token> & tokens,
        server_token_dump_kind kind) const {
    if (!impl_) {
        return;
    }

    std::string error;
    if (!impl_->append(tokens, kind, &error)) {
        SRV_ERR("request token dump failed: %s\n", error.c_str());
    }
}

