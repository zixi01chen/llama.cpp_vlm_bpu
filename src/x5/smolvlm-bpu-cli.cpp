#include "arg.h"
#include "base64.hpp"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"
#include "ggml.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef NDEBUG
#include "ggml-alloc.h"
#include "ggml-backend.h"
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_status.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <src/llama-context.h>
#include <chrono>

#define ALIGNED_2E(w, alignment) \
  ((static_cast<uint32_t>(w) + (alignment - 1U)) & (~(alignment - 1U)))
#define ALIGN_16(w) ALIGNED_2E(w, 16U)

// 使用 SmolVLM2 的预处理参数的 GetBGRTensorFromBGRImg 函数
int GetBGRTensorFromBGRImg(const cv::Mat &bgr_mat_tmp,
                            cv::Mat &pixel_values_mat,
                            int scaled_img_height, 
                            int scaled_img_width) {
    cv::Mat bgr_mat;
    bgr_mat_tmp.copyTo(bgr_mat);
    auto w_stride = ALIGN_16(scaled_img_width);
    int channel = 3;
    int original_img_width = bgr_mat.cols;
    int original_img_height = bgr_mat.rows;
    cv::Mat pad_frame;
    
    if (static_cast<uint32_t>(original_img_width) != w_stride ||
      original_img_height != scaled_img_height) {
      pad_frame = cv::Mat(scaled_img_height, w_stride, CV_8UC3, cv::Scalar::all(0));
      
      if (static_cast<uint32_t>(original_img_width) > w_stride ||
        original_img_height > scaled_img_height) {
        float ratio_w = static_cast<float>(original_img_width) / static_cast<float>(w_stride);
        float ratio_h = static_cast<float>(original_img_height) / static_cast<float>(scaled_img_height);
        float dst_ratio = std::max(ratio_w, ratio_h);
        uint32_t resized_width = static_cast<float>(original_img_width) / dst_ratio;
        uint32_t resized_height = static_cast<float>(original_img_height) / dst_ratio;
        cv::resize(bgr_mat, bgr_mat, cv::Size(resized_width, resized_height));
        std::cout << "reize height: " << resized_height << "resize width: " << resized_width << std::endl;
      }
      
      // 复制到目标图像到起点
      bgr_mat.copyTo(pad_frame(cv::Rect(0, 0, bgr_mat.cols, bgr_mat.rows)));
    } else {
      pad_frame = bgr_mat;
    }
    
    cv::Mat mat_tmp_tmp;
    pad_frame.convertTo(mat_tmp_tmp, CV_32F); 
    cv::cvtColor(mat_tmp_tmp, mat_tmp_tmp, cv::COLOR_BGR2RGB);
    mat_tmp_tmp /= 255.0;
    
    // SmolVLM2 使用的均值和标准差都是 0.5
    cv::Scalar mean(0.5, 0.5, 0.5);  
    cv::Scalar std(0.5, 0.5, 0.5);   
    
    // 按通道减去均值,再除以标准差
    std::vector<cv::Mat> channels(3);
    cv::split(mat_tmp_tmp, channels);  // 分离通道
    for (int i = 0; i < 3; i++) {
      channels[i] = (channels[i] - mean[i]) / std[i];
    }
    
    // 合并通道
    cv::merge(channels, pixel_values_mat);
    return 0;
}

// smolvlm2_vit_model_infer 函数，添加了耗时统计
struct llava_image_embed * smolvlm2_vit_model_infer(hbPackedDNNHandle_t& packed_dnn_handle, const std::string & fname) {
    // 开始计时
    auto total_start = std::chrono::high_resolution_clock::now();
    
    llava_image_embed * embed = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    
    // 第二步获取模型名称
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
    
    // 第三步获取dnn_handle
    hbDNNHandle_t dnn_handle;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    
    // 第四步准备输入数据
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    
    hbDNNTensor input;
    hbDNNTensorProperties input_properties;
    hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0);
    
    int model_input_width = 0;
    int model_input_height = 0;
    int channel = 3;
    int src_elem_size = 4;
    
    if (input_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        model_input_height = input_properties.alignedShape.dimensionSize[2];
        model_input_width = input_properties.alignedShape.dimensionSize[3];
    } else if (input_properties.tensorLayout == HB_DNN_LAYOUT_NHWC) {
        model_input_height = input_properties.alignedShape.dimensionSize[1];
        model_input_width = input_properties.alignedShape.dimensionSize[2];
    }
    
    input.properties = input_properties;
    auto &mem = input.sysMem[0];
    
    size_t size = model_input_height * model_input_width * channel * src_elem_size;
    cv::Mat bgr_mat = cv::imread(fname, cv::IMREAD_COLOR);
    cv::Mat pixel_mat;
    GetBGRTensorFromBGRImg(bgr_mat, pixel_mat, model_input_height, model_input_width);
    uint8_t *pixel_data = pixel_mat.data;
    
    hbSysAllocCachedMem(&mem, size);    
    auto *hb_mem_addr = reinterpret_cast<uint8_t *>(mem.virAddr);
    
    if (input_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        for (int h = 0; h < model_input_height; ++h) {
            for (int w = 0; w < model_input_width; ++w) {
                for (int c = 0; c < channel; ++c) {
                    auto *raw = hb_mem_addr + c * model_input_height * model_input_width * src_elem_size + h * model_input_width * src_elem_size + w * src_elem_size;
                    auto *src = pixel_data + h * model_input_width * channel * src_elem_size + w * channel * src_elem_size + c * src_elem_size;
                    memcpy(raw, src, src_elem_size);
                }
            }
        }
    } else {
        for (int h = 0; h < model_input_height; ++h) {
            auto *raw = hb_mem_addr + h * model_input_width * channel * src_elem_size;
            auto *src = pixel_data + h * model_input_width * channel * src_elem_size;
            memcpy(raw, src, model_input_width * channel * src_elem_size);
        }
    }
    
    hbSysFlushMem(&mem, HB_SYS_MEM_CACHE_CLEAN);
    
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count();
    
    // 第五步准备模型输出数据的空间
    int output_count;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    hbDNNTensor *output = new hbDNNTensor[output_count];
    
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        hbSysAllocCachedMem(&mem, out_aligned_size);
    }
    
    // 第六步推理模型 - BPU推理计时
    auto bpu_infer_start = std::chrono::high_resolution_clock::now();
    
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    hbDNNInfer(&task_handle,
                &output,
                &input,
                dnn_handle,
                &infer_ctrl_param);
    
    // 第七步等待任务结束
    hbDNNWaitTaskDone(task_handle, 0);
    
    auto bpu_infer_end = std::chrono::high_resolution_clock::now();
    auto bpu_infer_time = std::chrono::duration_cast<std::chrono::microseconds>(bpu_infer_end - bpu_infer_start).count();
    
    // 第八步解析模型输出
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    
    hbSysFlushMem(&(output->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
    hbDNNTensorProperties output_properties;
    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, 0);
    
    int num_tensors = 0;
    int length_tensors = 0;
    
    if (output_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        num_tensors = output_properties.alignedShape.dimensionSize[1];
        length_tensors = output_properties.alignedShape.dimensionSize[2];
    } else {
        LOG_ERR("%s: failed to get embedding\n", __func__);
        exit(1);
    }
    
    uint8_t* data = reinterpret_cast<uint8_t*>(output->sysMem[0].virAddr);
    size = num_tensors * length_tensors * sizeof(float);
    float* image_embed = static_cast<float*>(malloc(size));
    if (!image_embed) {
        LOG_ERR("%s: failed to alloc mem\n", __func__);
    }
    memcpy(image_embed, data, size);
    
    embed->embed = image_embed;
    embed->n_image_pos = num_tensors;
    
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    auto postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start).count();
    
    // 总耗时
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    
    // 打印详细的耗时统计
    LOG_INF("\n========== SmolVLM2 Vision Encoder (BPU) Performance ==========\n");
    LOG_INF("Image preprocessing time:    %8.3f ms\n", preprocess_time / 1000.0);
    LOG_INF("BPU inference time:         %8.3f ms  <-- Vision Encoder on BPU\n", bpu_infer_time / 1000.0);
    LOG_INF("Output postprocessing time: %8.3f ms\n", postprocess_time / 1000.0);
    LOG_INF("--------------------------------------------------------\n");
    LOG_INF("Total vision encoding time: %8.3f ms\n", total_time / 1000.0);
    LOG_INF("Output shape: [%d, %d] (n_image_pos=%d)\n", num_tensors, length_tensors, num_tensors);
    LOG_INF("===============================================================\n\n");
    
    // 释放任务
    hbDNNReleaseTask(task_handle);
    // 释放内存
    hbSysFreeMem(&(input.sysMem[0]));
    hbSysFreeMem(&(output->sysMem[0]));
    // 释放模型
    hbDNNRelease(packed_dnn_handle);
    
    return embed;
}

// 修改后的 smolvlm2_eval_image_embed 函数
static bool smolvlm2_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed,
                                     int n_batch, int * n_past, int * st_pos_id) {
    int n_embd = llama_model_n_embd(llama_get_model(ctx_llama));
    auto img_tokens = image_embed->n_image_pos;  // 64 for SmolVLM2
    
    // SmolVLM2: 简单地按顺序处理视觉嵌入，不需要复杂的 mRoPE
    for (int i = 0; i < img_tokens; i += n_batch) {
        int n_eval = img_tokens - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        
        // 为当前批次准备位置数据
        std::vector<llama_pos> batch_pos(n_eval);
        for (int j = 0; j < n_eval; j++) {
            batch_pos[j] = *st_pos_id + j;
        }
        
        // 构造批次
        llama_batch batch = {
            int32_t(n_eval),                      // n_tokens
            nullptr,                              // token
            (image_embed->embed + i * n_embd),    // embed
            batch_pos.data(),                     // pos
            nullptr,                              // n_seq_id
            nullptr,                              // seq_id
            nullptr,                              // logits
        };
        
        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        
        *n_past += n_eval;
        *st_pos_id += n_eval;
    }
    
    return true;
}

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past, int * st_pos_id) {
    int N = (int) tokens.size();
    std::vector<llama_pos> pos;
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        auto batch = llama_batch_get_one(&tokens[i], n_eval);
        // TODO: add mrope pos ids somewhere else
        pos.resize(batch.n_tokens * 4);
        std::fill(pos.begin(), pos.end(), 0);
        for (int j = 0; j < batch.n_tokens * 3; j ++) {
            pos[j] = *st_pos_id + (j % batch.n_tokens);
        }
        batch.pos = pos.data();
        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
        *st_pos_id += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past, int * st_pos_id) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past, st_pos_id);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, int * st_pos_id, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = common_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past, st_pos_id);
    return true;
}

static const char * sample(struct common_sampler * smpl,
                           struct llama_context * ctx_llama,
                           int * n_past, int * st_pos_id) {
    const llama_token id = common_sampler_sample(smpl, ctx_llama, -1);
    common_sampler_accept(smpl, id, true);
    const llama_model * model = llama_get_model(ctx_llama);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    static std::string ret;
    if (llama_vocab_is_eog(vocab, id)) {
        ret = "</s>";
    } else {
        ret = common_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past, st_pos_id);
    return ret.c_str();
}

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}

static bool prompt_contains_image(const std::string& prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void print_usage(int, char ** argv) {
    LOG("\n example usage:\n");
    LOG("\n     %s -m <SmolVLM2-500M-Video-Instruct-Q8_0.gguf> --mmproj <siglip_model_int16_SmolVLM2_500M_Instruct_X5.bin> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.4 is recommended for better quality.\n");
}

static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, 
                          common_params * params, const std::string & prompt) {
    int n_past = 0;
    int cur_pos_id = 0;
    
    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;
    
    // 处理用户输入
    std::string user_input = prompt;
    if (user_input.empty()) {
        user_input = "describe the image in detail.";
    }
    
    // SmolVLM2 官方模板格式（简化版，单图对话）
    // <|im_start|>User: <image>{user_input}<end_of_utterance>\nAssistant:
    std::string full_prompt = "<|im_start|>User: ";
    std::string image_placeholder = "<image>";
    std::string user_message_end = user_input + "<end_of_utterance>\nAssistant:";
    
    LOG_INF("=== Full conversation ===\n");
    LOG_INF("Part 1: %s\n", full_prompt.c_str());
    LOG_INF("Part 2: [IMAGE EMBEDDING HERE]\n");
    LOG_INF("Part 3: %s\n", user_message_end.c_str());
    
    if (params->verbose_prompt) {
        LOG_INF("\n=== Tokenization details ===\n");
        
        LOG_INF("Part 1 tokens:\n");
        auto tmp = common_tokenize(ctx_llava->ctx_llama, full_prompt, true, true);
        for (int i = 0; i < (int) tmp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
        }
        
        LOG_INF("\nPart 3 tokens:\n");
        tmp = common_tokenize(ctx_llava->ctx_llama, user_message_end, false, true);
        for (int i = 0; i < (int) tmp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
        }
    }
    
    // 1. 评估开始部分 "<|im_start|>User: "
    eval_string(ctx_llava->ctx_llama, full_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, true);
    
    // 2. 插入图像嵌入（对应 <image> 占位符）
    if (image_embed != nullptr) {
        LOG_INF("\nInserting image embedding at position %d\n", cur_pos_id);
        smolvlm2_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past, &cur_pos_id);
    }
    
    // 3. 评估用户输入和助手开始标记
    eval_string(ctx_llava->ctx_llama, user_message_end.c_str(), params->n_batch, &n_past, &cur_pos_id, false);
    
    // 生成响应
    LOG("\n=== Generating response ===\n");
    
    struct common_sampler * smpl = common_sampler_init(ctx_llava->model, params->sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }
    
    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past, &cur_pos_id);
        response += tmp;
        
        // SmolVLM2 的停止条件（保持原有的截断逻辑）
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(response.c_str(), "<end_of_utterance>")) break;  // 官方结束标记
        if (strstr(response.c_str(), "<|im_start|>")) break;  // 防止生成新对话
        if (strstr(response.c_str(), "User:")) break;  // 防止生成新的用户输入
        if (strstr(response.c_str(), "Assistant:")) break;  // 防止重复助手标记
        
        LOG("%s", tmp);
        fflush(stdout);
    }
    
    common_sampler_free(smpl);
    LOG("\n");
}

static struct llama_model * llava_init(common_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = common_model_params_to_llama(*params);

    llama_model * model = llama_model_load_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

static struct llava_context * llava_init_context(common_params * params, llama_model * model, hbPackedDNNHandle_t& packed_dnn_handle) {
    const char * model_file_name = params->mmproj.c_str();
    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1);

    llama_context_params ctx_params = common_context_params_to_llama(*params);
    ctx_params.n_ctx           = params->n_ctx < 2048 ? 2048 : params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_init_from_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_ERR("%s: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto * ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));
    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->model = model;
    return ctx_llava;
}

static void llava_free(struct llava_context * ctx_llava) {
    llama_free(ctx_llava->ctx_llama);
    llama_model_free(ctx_llava->model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LLAVA, print_usage)) {
        return 1;
    }

    common_init();

    if (params.mmproj.empty() || (params.image.empty() && !prompt_contains_image(params.prompt))) {
        print_usage(argc, argv);
        return 1;
    }
    
    auto * model = llava_init(&params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init llava model\n", __func__);
        return 1;
    }

    if (prompt_contains_image(params.prompt)) {
        hbPackedDNNHandle_t packed_dnn_handle;
        auto * ctx_llava = llava_init_context(&params, model, packed_dnn_handle);
        llava_image_embed * image_embed = smolvlm2_vit_model_infer(packed_dnn_handle, params.prompt);

        // process the prompt
        process_prompt(ctx_llava, image_embed, &params, params.prompt);
        llama_perf_context_print(ctx_llava->ctx_llama);
        llava_image_embed_free(image_embed);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    } else {
        for (auto & image : params.image) {
            hbPackedDNNHandle_t packed_dnn_handle;
            auto * ctx_llava = llava_init_context(&params, model, packed_dnn_handle);

            llava_image_embed * image_embed = smolvlm2_vit_model_infer(packed_dnn_handle, image);
            if (!image_embed) {
                LOG_ERR("%s: failed to load image %s. Terminating\n\n", __func__, image.c_str());
                return 1;
            }

            // process the prompt
            process_prompt(ctx_llava, image_embed, &params, params.prompt);
            llama_perf_context_print(ctx_llava->ctx_llama);
            llava_image_embed_free(image_embed);
            ctx_llava->model = NULL;
            llava_free(ctx_llava);
        }
    }

    llama_model_free(model);

    return 0;
}
