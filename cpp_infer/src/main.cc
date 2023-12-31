
#include <sys/time.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "common_inc/infer.h"

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(input0_path, ".", "input0 path");
DEFINE_string(dataset_name, "widerface", "dataset name");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_type, "CPU", "device type");

int predict_widerface(std::string input_path, std::vector<MSTensor> &model_inputs, Model *model,
                      std::map<double, double> &costTime_map) {
  auto input0_files = GetAllFiles(input_path);
  if (input0_files.empty()) {
    std::cout << "ERROR: no input data." << std::endl;
    return 1;
  }
  size_t size = input0_files.size();
  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    std::cout << "Start predict input files:" << input0_files[i] <<std::endl;
    auto input0 = ReadFileToTensor(input0_files[i]);
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        input0.Data().get(), input0.DataSize());

    gettimeofday(&start, nullptr);
    Status ret = model->Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << input0_files[i] << " failed." << std::endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    int rst = WriteResult(input0_files[i], outputs);
    if (rst != 0) {
        std::cout << "write result failed." << std::endl;
        return rst;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }
  std::vector<MSTensor> model_inputs = model.GetInputs();

  std::map<double, double> costTime_map;

  if (FLAGS_dataset_name == "widerface") {
    int ret = predict_widerface(FLAGS_input0_path, model_inputs, &model, costTime_map);
    if (ret != 0) {
      return ret;
    }
  } else {
    auto input0_files = GetAllInputData(FLAGS_input0_path);
    if (input0_files.empty()) {
      std::cout << "ERROR: no input data." << std::endl;
      return 1;
    }
    size_t size = input0_files.size();
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < input0_files[i].size(); ++j) {
        struct timeval start = {0};
        struct timeval end = {0};
        double startTimeMs;
        double endTimeMs;
        std::vector<MSTensor> inputs;
        std::vector<MSTensor> outputs;
        std::cout << "Start predict input files:" << input0_files[i][j] <<std::endl;
        auto decode = Decode();
        auto resize = Resize({256, 256});
        auto centercrop = CenterCrop({224, 224});
        auto normalize = Normalize({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375});
        auto hwc2chw = HWC2CHW();

        Execute SingleOp({decode, resize, centercrop, normalize, hwc2chw});
        auto imgDvpp = std::make_shared<MSTensor>();
        SingleOp(ReadFileToTensor(input0_files[i][j]), imgDvpp.get());
        inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                            imgDvpp->Data().get(), imgDvpp->DataSize());
      gettimeofday(&start, nullptr);
      Status ret = model.Predict(inputs, &outputs);
      gettimeofday(&end, nullptr);
      if (ret != kSuccess) {
        std::cout << "Predict " << input0_files[i][j] << " failed." << std::endl;
        return 1;
      }
      startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
      endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
      costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
      int rst = WriteResult(input0_files[i][j], outputs);
      if (rst != 0) {
          std::cout << "write result failed." << std::endl;
          return rst;
      }
    }
    }
  }

  double average = 0.0;
  int inferCount = 0;

  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = iter->second - iter->first;
    average += diff;
    inferCount++;
  }
  average = average / inferCount;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << inferCount << std::endl;
  std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << inferCount << std::endl;
  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}
