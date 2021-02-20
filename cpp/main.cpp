// https://pytorch.org/cppdocs/frontend.html

#include <torch/torch.h>
#include <iostream>


struct Net : torch::nn::Module {
    Net() {
        l1 = register_module("l1", torch::nn::Linear(784, 256));
        l2 = register_module("l2", torch::nn::Linear(256, 64));
        l3 = register_module("l3", torch::nn::Linear(64, 10));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(l1->forward(x.reshape({x.size(0), 784})));
        x = torch::relu(l2->forward(x));
        x = torch::log_softmax(l3->forward(x), 0);
        return x;
    }
    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
};


int main() {
    std::cout << "Hello!" << std::endl;
    
    auto net = std::make_shared<Net>();
    
    // Create a multi-threaded data loader for the MNIST dataset.
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/64);
    
    torch::optim::SGD optimizer(net->parameters(), 0.01);
    
    for (size_t epoch = 0; epoch < 10; epoch++) {
        size_t batch_index = 0;
        for (auto& batch : *data_loader) {
            optimizer.zero_grad();
            torch::Tensor prediction = net->forward(batch.data);
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
            loss.backward();
            optimizer.step();
            
            if (++batch_index % 100 == 0) {
                    std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                              << " | Loss: " << loss.item<float>() << std::endl;
            }
        }
    }
}

