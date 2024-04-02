// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "yoloface.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

static inline float sigmoid(float x){
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

std::vector<float> softmax(const std::vector<float> & input)
{
    float total = 0.;
    for(auto x : input)
    {
        total += exp(x);
    }
    std::vector<float> result;
    for(auto x : input)
    {
        result.push_back(exp(x) / total);
    }
    return result;
}

bool cmp(Object b1, Object b2) {
    return b1.prob > b2.prob;
}

void nms(std::vector<Object> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

std::vector<float> LetterboxImage(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size)
{
    auto in_h 	= static_cast<float>(src.rows);
    auto in_w	= static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h 	= static_cast<int>(in_h * scale);
    int mid_w 	= static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h), 0, 0, cv::INTER_NEAREST);

    int top 	= (static_cast<int>(out_h) - mid_h) 	/ 2;
    int down 	= (static_cast<int>(out_h) - mid_h + 1) / 2;
    int left 	= (static_cast<int>(out_w) - mid_w) 	/ 2;
    int right 	= (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}

void decode(const ncnn::Mat& feat_blob, std::vector<Object> &prebox, float threshold, int stride)
{
    int reg_max         = 16;
    int fea_h 			= feat_blob.h;
    int fea_w 			= feat_blob.w;
    int spacial_size	= fea_w * fea_h;
    float *ptr_b   		= (float*)(feat_blob.data);
    float *ptr_c        = ptr_b + spacial_size * reg_max * 4;
    float *ptr_p        = ptr_c + spacial_size;

    for(int i = 0; i < fea_h; i++)
    {
        for(int j = 0; j < fea_w; j++)
        {
            int index = i * fea_w + j;
            float box_prob 	= sigmoid(ptr_c[index]);
            if(box_prob > threshold)
            {
                float pred_ltrb[4];
                std::vector<float> dfl_value;
                std::vector<float> dfl_softmax;
                dfl_value.resize(reg_max);
                dfl_softmax.resize(reg_max);
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    for(int n=0; n < reg_max; n++){
                        dfl_value[n]=ptr_b[index + (reg_max * k + n) * spacial_size];
                    }

                    dfl_softmax = softmax(dfl_value);
                    for (int l = 0; l < reg_max; l++){
                        dis += l * dfl_softmax[l];
                    }
                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (j + 0.5f) * stride;
                float pb_cy = (i + 0.5f) * stride;

                float x1 = pb_cx - pred_ltrb[0];
                float y1 = pb_cy - pred_ltrb[1];
                float x2 = pb_cx + pred_ltrb[2];
                float y2 = pb_cy + pred_ltrb[3];

                Object temp_box;
                temp_box.prob 	= box_prob;
                temp_box.label 	= 0;
                temp_box.x1 	= x1;
                temp_box.y1 	= y1;
                temp_box.x2 	= x2;
                temp_box.y2 	= y2;

                temp_box.pts.resize(5);
                for(int l = 0; l < 5; l++)
                {
                    temp_box.pts[l].x 	  = (ptr_p[(l * 3 + 0) * spacial_size + index] * 2 + j) * stride;
                    temp_box.pts[l].y 	  = (ptr_p[(l * 3 + 1) * spacial_size + index] * 2 + i) * stride;
                    temp_box.pts[l].score = sigmoid(ptr_p[(l * 3 + 2) * spacial_size + index]);
                }
                prebox.push_back(temp_box);
            }
        }
    }
}


YoloFace::YoloFace()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int YoloFace::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    yoloface.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yoloface.opt = ncnn::Option();
#if NCNN_VULKAN
    yoloface.opt.use_vulkan_compute = use_gpu;
#endif

    yoloface.opt.num_threads = ncnn::get_big_cpu_count();
    yoloface.opt.blob_allocator = &blob_pool_allocator;
    yoloface.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);
    yoloface.load_param(mgr, parampath);
    yoloface.load_model(mgr, modelpath);

    std::string type(modelpath);
    if(type.find("lite-t") == std::string::npos) lite_t = false;
    else lite_t = true;

    target_size = _target_size;
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}


int YoloFace::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    cv::Mat dst;
    target_size = 640;
    int img_w = rgb.cols;
    int img_h = rgb.rows;

    std::vector<float> infos = LetterboxImage(rgb, dst, cv::Size(target_size, target_size));
    ncnn::Mat in_pad = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_RGB, target_size, target_size);

    in_pad.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex = yoloface.create_extractor();
    ex.input("images", in_pad);

    // stride 8
    {
        ncnn::Mat out;
        ex.extract("output0", out);
        decode(out, objects, prob_threshold, 8);
    }

    // stride 16
    {
        ncnn::Mat out;
        if(lite_t) ex.extract("884", out);
        else ex.extract("1076", out);
        decode(out, objects, prob_threshold, 16);
    }

    // stride 32
    {
        ncnn::Mat out;
        if(lite_t) ex.extract("892", out);
        else ex.extract("1084", out);
        decode(out, objects, prob_threshold, 32);
    }

    std::sort(objects.begin(), objects.end(), cmp);
    nms(objects, nms_threshold);

    for(int i = 0; i < objects.size(); i ++)
    {
        objects[i].x1 = (objects[i].x1 - infos[0]) / infos[2];
        objects[i].y1 = (objects[i].y1 - infos[1]) / infos[2];
        objects[i].x2 = (objects[i].x2 - infos[0]) / infos[2];
        objects[i].y2 = (objects[i].y2 - infos[1]) / infos[2];
        // clip
        //objects[i].x1 = std::max(std::min(objects[i].x1, (float)(img_w - 1)), 0.f);
        //objects[i].y1 = std::max(std::min(objects[i].y1, (float)(img_h - 1)), 0.f);
        //objects[i].x2 = std::max(std::min(objects[i].x2, (float)(img_w - 1)), 0.f);
        //objects[i].y2 = std::max(std::min(objects[i].y2, (float)(img_h - 1)), 0.f);

        for(int j = 0; j < 5; j ++)
        {
            objects[i].pts[j].x = (objects[i].pts[j].x - infos[0]) / infos[2];
            objects[i].pts[j].y = (objects[i].pts[j].y - infos[1]) / infos[2];
        }
    }

    return 0;
}

int YoloFace::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{

    static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, cv::Point(obj.x1, obj.y1), cv::Point(obj.x2, obj.y2), cc, 2);

        char text[256];
        sprintf(text, "%.1f%%",  obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.x1;
        int y = obj.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
        for (int j = 0; j < obj.pts.size(); j++)
        {
            cv::circle(rgb, cv::Point(obj.pts[j].x,obj.pts[j].y), 2, cv::Scalar(0, 255, 0), -1);
        }
    }

    return 0;
}
