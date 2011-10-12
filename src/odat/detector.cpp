#include <boost/algorithm/string.hpp>

#include "odat/detector.h"


void odat::Detector::loadModelList(const std::string& models)
{
  std::vector<std::string> model_list;
  boost::split(model_list, models, boost::is_any_of(", \t"));
  std::vector<std::string> filtered_list;
  for (size_t i = 0; i < model_list.size(); ++i)
  {
    if (model_list[i].length() != 0)
      filtered_list.push_back(model_list[i]);
  }
  loadModels(filtered_list);
}

void odat::Detector::loadAllModels()
{
  if (model_storage_ != NULL) {
    std::vector<std::string> models;
    model_storage_->getModelList(getName(), models);
    loadModels(models);
  }
}
  
