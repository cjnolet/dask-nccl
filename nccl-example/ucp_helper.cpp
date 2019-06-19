#include <ucp/api/ucp.h>
#include "ucp_helper.h"

void call_ucp_function(void *worker) {
    
    ucp_worker_print_info((ucp_worker_h)worker, stdout);
}