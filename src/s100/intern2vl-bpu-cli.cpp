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

#include "dnn/hb_dnn.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <src/llama-context.h>

#define ALIGNED_2E(w, alignment) \
  ((static_cast<uint32_t>(w) + (alignment - 1U)) & (~(alignment - 1U)))
#define ALIGN_16(w) ALIGNED_2E(w, 16U)

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
      }
      // 复制到目标图像到起点
      bgr_mat.copyTo(pad_frame(cv::Rect(0, 0, bgr_mat.cols, bgr_mat.rows)));
    } else {
      pad_frame = bgr_mat;
    }

    cv::Mat mat_tmp_tmp;
    pad_frame.convertTo(mat_tmp_tmp, CV_32F); 
    mat_tmp_tmp /= 255.0;

    cv::Scalar mean(0.485, 0.456, 0.406);  // BGR 通道均值
    cv::Scalar std(0.229, 0.224, 0.225);   // BGR 通道标准差

    // 按通道减去均值，再除以标准差
    std::vector<cv::Mat> channels(3);
    cv::split(mat_tmp_tmp, channels);  // 分离通道

    for (int i = 0; i < 3; i++) {
      channels[i] = (channels[i] - mean[i]) / std[i];
    }

    // 合并通道
    cv::merge(channels, pixel_values_mat);
    return 0;
}

struct llava_image_embed * internvl2_vit_model_infer(hbDNNPackedHandle_t& packed_dnn_handle, const std::string & fname) {

    llava_image_embed * embed = (llava_image_embed*)malloc(sizeof(llava_image_embed));

    // 第二步获取模型名称
    const char **model_name_list;
    int model_count = 0;
    hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);

    // 第三步获取dnn_handle
    hbDNNHandle_t dnn_handle;
    hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

    // 第四步准备输入数据
    std::vector<hbDNNTensor> input(1);
    hbDNNTensorProperties input_properties;
    hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0);
    int channel = 3;
    int src_elem_size = 4;
    int model_input_height = input_properties.validShape.dimensionSize[2];
    int model_input_width = input_properties.validShape.dimensionSize[3];
    input[0].properties = input_properties;
    auto &mem = input[0].sysMem;
    
    size_t size = model_input_height * model_input_width * channel * src_elem_size;
    cv::Mat bgr_mat = cv::imread(fname, cv::IMREAD_COLOR);
    cv::Mat pixel_mat;
    GetBGRTensorFromBGRImg(bgr_mat, pixel_mat, model_input_height, model_input_width);
    uint8_t *pixel_data = pixel_mat.data;
    
    hbUCPMallocCached(&mem, size, 0);
    auto *hb_mem_addr = reinterpret_cast<uint8_t *>(mem.virAddr);
    for (int h = 0; h < model_input_height; ++h) {
    for (int w = 0; w < model_input_width; ++w) {
        for (int c = 0; c < channel; ++c) {
        auto *raw = hb_mem_addr + c * model_input_height * model_input_width * src_elem_size + h * model_input_width * src_elem_size + w * src_elem_size;
        auto *src = pixel_data + h * model_input_width * channel * src_elem_size + w * channel * src_elem_size + c * src_elem_size;
        memcpy(raw, src, src_elem_size);
        }
    }
    }
    hbUCPMemFlush(&mem, HB_SYS_MEM_CACHE_CLEAN);

    // 第五步准备模型输出数据的空间
    int output_count;
    hbDNNGetOutputCount(&output_count, dnn_handle);
    std::vector<hbDNNTensor> output(output_count);
    for (int i = 0; i < output_count; i++) {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbUCPSysMem &mem = output[i].sysMem;
        hbUCPMallocCached(&mem, out_aligned_size, 0);
    }

    // 第六步推理模型
    hbUCPTaskHandle_t task_handle = nullptr;
    hbDNNInferV2(&task_handle,
                output.data(),
                input.data(),
                dnn_handle);
    hbUCPSchedParam infer_sched_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&infer_sched_param);
    hbUCPSubmitTask(task_handle, &infer_sched_param);
              
    // 第七步等待任务结束
    hbUCPWaitTaskDone(task_handle, 0);
    //第八步解析模型输出，例子就获取mobilenetv1的top1分类
    hbUCPMemFlush(&(output[0].sysMem), HB_SYS_MEM_CACHE_INVALIDATE);

    hbDNNTensorProperties output_properties;
    hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, 0);
    int num_tensors = output_properties.validShape.dimensionSize[1];
    int length_tensors = output_properties.validShape.dimensionSize[2];

    uint8_t* data = reinterpret_cast<uint8_t*>(output[0].sysMem.virAddr);
    size = num_tensors * length_tensors * sizeof(float);

    float* image_embed = static_cast<float*>(malloc(size));
    if (!image_embed) {
        LOG_ERR("%s: failed to alloc mem\n", __func__);
    }

    memcpy(image_embed, data, size);

    embed->embed = image_embed;
    embed->n_image_pos = num_tensors;

    // 释放任务
    hbUCPReleaseTask(task_handle);

    // 释放内存
    hbUCPFree(&(input[0].sysMem));
    hbUCPFree(&(output[0].sysMem));

    // 释放模型
    hbDNNRelease(packed_dnn_handle);

    return embed;
}

static bool internvl2_eval_image_embed(llama_context * ctx_llama, const struct llava_image_embed * image_embed,
                                     int n_batch, int * n_past, int * st_pos_id) {
    int n_embd  = llama_model_n_embd(llama_get_model(ctx_llama));
    const int patch_size = 14 * 2;
    int ph = 16;
    int pw = 16;
    auto img_tokens = image_embed->n_image_pos;
    
    // llama_pos mrope_pos[img_tokens * 4];
    std::vector<llama_pos> mrope_pos;
    mrope_pos.resize(img_tokens * 4);

    for (int y = 0; y < ph; y++)
    {
        for (int x = 0; x < pw; x++)
        {
            int i = y * pw + x;
            mrope_pos[i] = *st_pos_id;
            mrope_pos[i + img_tokens] = *st_pos_id + y;
            mrope_pos[i + img_tokens * 2] = *st_pos_id + x;
            mrope_pos[i + img_tokens * 3] = 0;
        }
    }
    *st_pos_id += std::max(pw, ph);

    int processed = 0;
    std::vector<llama_pos> batch_mrope_pos;
    batch_mrope_pos.resize(img_tokens * 4);

    for (int i = 0; i < img_tokens; i += n_batch) {
        int n_eval = img_tokens - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }

        // llama_pos batch_mrope_pos[n_eval * 4];
        std::fill(batch_mrope_pos.begin(), batch_mrope_pos.end(), 0);
        memcpy(batch_mrope_pos.data(), &mrope_pos[processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 1], &mrope_pos[img_tokens * 1 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 2], &mrope_pos[img_tokens * 2 + processed], n_eval * sizeof(llama_pos));
        memcpy(&batch_mrope_pos[n_eval * 3], &mrope_pos[img_tokens * 3 + processed], n_eval * sizeof(llama_pos));

        llama_batch batch = {
            int32_t(n_eval),                // n_tokens
            nullptr,                        // token
            (image_embed->embed+i*n_embd),  // embed
            batch_mrope_pos.data(),         // pos
            nullptr,  // n_seq_id
            nullptr,  // seq_id
            nullptr,  // logits
        };

        if (llama_decode(ctx_llama, batch)) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
        processed += n_eval;
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
    LOG("\n     %s -m <Qwen2.5-0.5B-Instruct-F16.gguf> --mmproj <vit_model_int16.bin> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG("\n note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static void process_prompt(struct llava_context * ctx_llava, struct llava_image_embed * image_embed, common_params * params, const std::string & prompt) {
    int n_past = 0;
    int cur_pos_id = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<|vision_start|>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<|vision_pad|>").length());
        LOG_INF("system_prompt: %s\n", system_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
        LOG_INF("user_prompt: %s\n", user_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    } else {
        // come inside
        // llava-1.5 native mode
        LOG_INF("prompt: %s\n", prompt.c_str());
        system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>";
        user_prompt = "<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        if (params->verbose_prompt) {
            auto tmp = common_tokenize(ctx_llava->ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx_llava->ctx_llama, tmp[i]).c_str());
            }
        }
    }

    eval_string(ctx_llava->ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, true);
    if (image_embed != nullptr) {
        internvl2_eval_image_embed(ctx_llava->ctx_llama, image_embed, params->n_batch, &n_past, &cur_pos_id);
    }

    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, &cur_pos_id, false);

    // generate the response
    LOG("\n");

    struct common_sampler * smpl = common_sampler_init(ctx_llava->model, params->sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    std::string response = "";
    for (int i = 0; i < max_tgt_len; i++) {
        // 在这里开始融合特征、处理融合特征。
        const char * tmp = sample(smpl, ctx_llava->ctx_llama, &n_past, &cur_pos_id);
        response += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(tmp, "###")) break; // Yi-VL behavior
        LOG("%s", tmp);
        if (strstr(response.c_str(), "<|im_end|>")) break; // Yi-34B llava-1.6 - for some reason those decode not as the correct token (tokenizer works)
        if (strstr(response.c_str(), "<|im_start|>")) break; // Yi-34B llava-1.6
        if (strstr(response.c_str(), "USER:")) break; // mistral llava-1.6

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

static struct llava_context * llava_init_context(common_params * params, llama_model * model, hbDNNPackedHandle_t& packed_dnn_handle) {
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
        hbDNNPackedHandle_t packed_dnn_handle;
        auto * ctx_llava = llava_init_context(&params, model, packed_dnn_handle);

        llava_image_embed * image_embed = internvl2_vit_model_infer(packed_dnn_handle, params.prompt);

        // process the prompt
        process_prompt(ctx_llava, image_embed, &params, params.prompt);

        llama_perf_context_print(ctx_llava->ctx_llama);
        llava_image_embed_free(image_embed);
        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    } else {
        for (auto & image : params.image) {
            hbDNNPackedHandle_t packed_dnn_handle;
            auto * ctx_llava = llava_init_context(&params, model, packed_dnn_handle);

            llava_image_embed * image_embed = internvl2_vit_model_infer(packed_dnn_handle, image);

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
