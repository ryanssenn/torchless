#include "../../include/parameters.h"
#include <iostream>

#include <fcntl.h>     // declares open()
#include <unistd.h>    // declares close()
#include <sys/mman.h>  // declares mmap()
#include <sys/stat.h>

// Memory-map the model file into virtual address space and return base pointer.
void* Parameters::map_file(int fd){
    // Get the size of the binary file
    struct stat st;
    if (fstat(fd, &st) < 0){
        std::cerr << "Model binary get size failed" << std::endl;
        std::exit(1);
    }

    size_t size = st.st_size;

    // Load the file into virtual memory
    void* p = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (p == MAP_FAILED){
        std::cerr << "Model mmap failed" << std::endl;
        std::exit(1);
    }

    return p;
}

// Extract model hyperparameters from JSON header into a Config struct.
void Parameters::load_config(nlohmann::json& header){
    nlohmann::json m = header["metadata"];

    config.hidden_size       = std::stoi(m["hidden_size"].get<std::string>());
    config.intermediate_size = std::stoi(m["intermediate_size"].get<std::string>());
    config.n_layers          = std::stoi(m["n_layers"].get<std::string>());
    config.n_heads           = std::stoi(m["n_heads"].get<std::string>());
    config.n_kv_heads        = std::stoi(m["n_kv_heads"].get<std::string>());
    config.vocab_size        = std::stoi(m["vocab_size"].get<std::string>());
    config.sliding_window    = std::stoi(m["sliding_window"].get<std::string>());
    config.rope_theta        = std::stof(m["rope_theta"].get<std::string>());
    config.norm_eps          = std::stof(m["norm_eps"].get<std::string>());
    config.max_position_embeddings = std::stoi(m["max_position_embeddings"].get<std::string>());
    config.head_dim          = config.hidden_size / config.n_heads;
    config.quant             = m["quant"].get<std::string>();
}

void Parameters::load_tensor(std::unordered_map<std::string, std::variant<Tensor<float>, Tensor<int8_t>>>& m, char* p, const std::string& key, nlohmann::json& value){
    std::string type = value["dtype"];
    uint64_t offset = value["offset"];
    std::vector<size_t> shape = value["shape"];

    if (type == "f32"){
        Tensor<float> t = Tensor(reinterpret_cast<float*>(p + offset), shape);
        m.insert({key, t});
    }
    else if (type == "int8"){
        uint64_t scale_offset = value["scale_offset"];
        uint64_t scale_size = value["scale_size"];

        float* scale_start = reinterpret_cast<float*>(p + scale_offset);

        Tensor<int8_t> t = Tensor(reinterpret_cast<int8_t*>(p + offset), std::vector<float>(scale_start, scale_start+scale_size), shape);
        m.insert({key, t});
    }
}

// Load Tensor views for all weights using offsets from the JSON header.
void Parameters::load_weights(char* p, nlohmann::json& header){
    // Init empty maps per layer
    layer_weights.resize(config.n_layers);

    nlohmann::json t = header["tensors"];
    const std::string model_prefix = "model.layers.";

    for (auto& [key, value] : t.items()){

        // Layer-specific weight
        if (key.compare(0, model_prefix.size(), model_prefix) == 0){
            // Extract the layer number
            std::string rest = key.substr(model_prefix.size());
            int i = std::stoi(rest.substr(0, rest.find(".")));

            load_tensor(layer_weights[i], p, rest.substr(rest.find(".") + 1), value);
            continue;
        }

        // Otherwise global weight
        load_tensor(global_weights, p, key, value);
    }
}

void Parameters::load_parameters(const std::string& path){
    // Get file descriptor
    int fd = open(path.c_str(), O_RDONLY);

    if (fd < 0){
        std::cerr << "Model binary open failed" << std::endl;
        std::exit(1);
    }

    // Read the 8 byte header uint64_t size
    uint64_t header_size;
    read(fd, &header_size, sizeof(header_size));

    // Read and parse the JSON Header
    char* header = new char[header_size+1];
    read(fd, header, header_size);
    header[header_size] = '\0';
    nlohmann::json header_json = nlohmann::json::parse(std::string(header));

    load_config(header_json);
    tokenizer.load(header_json["tokenizer"]);

    // Map file into virtual memory
    void* p = map_file(fd);

    // The blob of tensors starts after header_size uint64 and the header
    char* start = static_cast<char*>(p) + header_size + sizeof(uint64_t);

    // Load tensor weights to pointers in mmap
    load_weights(start, header_json);

    delete[] header;

    close(fd);
}

template<typename T>
Tensor<T> Parameters::get_tensor(int layer, const std::string& name){
    if (layer == -1){
        for (auto& e : global_weights){
            if (name == e.first){
                return std::get<Tensor<T>>(e.second);
            }
        }

        assert(false && "tensor not found");
        return {};
    }

    for (auto& e : layer_weights[layer]){
        if (name == e.first){
            return std::get<Tensor<T>>(e.second);
        }
    }

    assert(false && "tensor not found");
    return {};
}

template Tensor<float> Parameters::get_tensor<float>(int, const std::string&);
template Tensor<int8_t> Parameters::get_tensor<int8_t>(int, const std::string&);


