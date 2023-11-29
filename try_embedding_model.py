from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel


# 模型名称
model_name = "voidful/albert_chinese_tiny"  # 这是一个轻量级的Chinese-ALBERT模型
# 加载预训练的Tokenizer和Model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, use_auth_token=False, trust_remote_code=True)
# 使用sentence-transformers的库来构建模型
word_embedding_model = models.Transformer(model_name, max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
# 组合模型
st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 编码句子
sentences = ["tiktok中的账号类型有哪几种？"]
sentence_embeddings = st_model.encode(sentences)

# 输出句子嵌入
print("Sentence Embedding:", sentence_embeddings[0])