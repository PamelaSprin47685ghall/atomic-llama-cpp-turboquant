#pragma once

#include "server-corpus-dump.h"

#include "../../common/common.h"
#include "../../tools/server/server-common.h"

#include <memory>
#include <vector>

class wanxiangqi_server_token_dump {
public:
    bool init(const common_params & params, bool is_resume);

    void append(
            const std::vector<llama_token> & tokens,
            server_token_dump_kind kind) const;

private:
    std::unique_ptr<server_token_dump> impl_;
};

