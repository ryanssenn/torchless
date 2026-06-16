#include "setup/context.h"
#include <iostream>

int main() {
    load_expected_values();

    int total   = 0;
    int failed  = 0;

    std::cout << "Running tests for model type: " << get_params()->config.quant << std::endl;

    for (const TestCase& t : tests) {
        if (t.quant == get_params()->config.quant){
            total++;
            std::cout << "Running " << t.name << std::endl;
            int result = t.func();
            if (result != 0) {
                failed++;
                std::cout << t.name << " has failed" << std::endl;
            }
        }
    }

    int passed = total - failed;
    std::cout << std::endl << "Summary " << ": " << passed << " / " << total << " tests passed" << std::endl;

    return 0;
}
