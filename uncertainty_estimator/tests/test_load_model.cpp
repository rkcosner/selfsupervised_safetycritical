//
// Created by ivan on 8/29/21.
//

#include <torch/script.h>
#include <experimental/filesystem>
#include <iostream>
#include <gtest/gtest.h>

torch::NoGradGuard no_grad;

class LoadModelTest : public ::testing::Test
{
 protected:
  LoadModelTest() :
    sample_model(torch::jit::load("/home/drew/Aeronvironment/catkin_ws/src/uncertainty_estimator/tests/data/test_model.th"))
  {
    sample_model.to(torch::kCPU);
    left = torch::rand({1, 200, 320});
    right = torch::rand({1, 200, 320});
    d3 = torch::rand({1, 200, 320});
    d3_hat = torch::rand({1,200,320});
    pred = torch::rand({1, 65, 200, 320});
//    torch::jit::script::Module container(torch::jit::load("data/test_container.th"));
//    left = container.attr("test_left").toTensor();
//    right = container.attr("test_right").toTensor();
//    d3 = container.attr("test_d3").toTensor();
//    dr_hat = container.attr("test_dr_hat").toTensor();
//    pred = container.attr("test_pred").toTensor();
  }

  torch::jit::script::Module sample_model;
  torch::Tensor left, right, d3, d3_hat, pred;
};

TEST_F(LoadModelTest, Load){
  std::cout << "std::filesystem::current_path():"  << std::endl << std::experimental::filesystem::current_path() << std::endl;
  for (const auto &item: sample_model.named_buffers()) {
    std::cout << "sample_model.named_buffers():"  << item.name << std::endl;
  }

  for (const auto &attribute: sample_model.named_attributes()) {
    std::cout << "attribute.name:"  << std::endl << attribute.name << std::endl;
  }
  for (const auto &method: sample_model.get_methods()) {
    std::cout << "method.name():"  << std::endl << method.name() << std::endl;
  }
}

TEST_F(LoadModelTest, ForwardTest) {
  auto images = torch::stack(std::vector<torch::Tensor>({left, right}), 1);
  std::cout << "images.sizes():"  << std::endl << images.sizes() << std::endl;
  torch::Dict<std::string, torch::Tensor> forward_input;
  forward_input.insert("images", images);
  auto model_output = sample_model.forward(std::vector<torch::jit::IValue>{forward_input});
  std::cout << "model_output.toTensor().sizes():"  << std::endl << model_output.toTensor().sizes() << std::endl;
}

TEST_F(LoadModelTest, ForwardCudaTest) {
  auto images = torch::stack(std::vector<torch::Tensor>({left, right}), 1).to(torch::kCUDA);
  std::cout << "images.sizes():"  << std::endl << images.sizes() << std::endl;
  torch::Dict<std::string, torch::Tensor> forward_input;
  forward_input.insert("images", images);
  sample_model.to(torch::kCUDA);
  auto model_output = sample_model.forward(std::vector<torch::jit::IValue>{forward_input});
  std::cout << "model_output.toTensor().sizes():"  << std::endl << model_output.toTensor().sizes() << std::endl;
}


