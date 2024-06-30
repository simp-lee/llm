package llm

import (
	"encoding/json"
	"fmt"
	"log/slog"
)

const (
	// query or document, default is document
	// 说明：文本转换为向量后可以应用于检索、聚类、分类等下游任务，
	// 对检索这类非对称任务为了达到更好的检索效果建议区分查询文本（query）和底库文本（document）类型,
	// 聚类、分类等对称任务可以不用特殊指定，采用系统默认值"document"即可
	EMBEDDING_DOCUMENT_TYPE = "document"
	EMBEDDING_QUERY_TYPE    = "query"
)

// AlibabaEmbeddingModelClient is a client to access the Alibaba Embedding API.
type AlibabaEmbeddingModelClient struct {
	BaseEmbeddingClient
}

// EmbeddingRequest is the request body for the Alibaba Embedding API.
type EmbeddingRequest struct {
	Model  string            `json:"model"`
	Input  EmbeddingInput    `json:"input"`
	Params map[string]string `json:"params"`
}

// EmbeddingInput is the input for the Alibaba Embedding API in the request body.
type EmbeddingInput struct {
	Texts []string `json:"texts"`
}

// EmbeddingResponse is the response body from the Alibaba Embedding API.
type EmbeddingResponse struct {
	Output struct {
		Embeddings []struct {
			TextIndex int       `json:"text_index"`
			Embedding []float64 `json:"embedding"`
		} `json:"embeddings"`
	} `json:"output"`

	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`

	RequestID string `json:"request_id"`
}

// GetEmbeddings sends a request to Alibaba DashScope API to get the embedding vectors for the input strings.
func (client *AlibabaEmbeddingModelClient) GetEmbeddings(inputs []string, embeddingType string) ([][]float64, error) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("Recovered from panic in AlibabaEmbeddingModelClient.GetEmbedding", slog.Any("error", r))
		}
	}()

	// Prepare the request body
	body := EmbeddingRequest{
		Model: client.ModelName,
		Input: EmbeddingInput{
			Texts: inputs,
		},
		Params: map[string]string{
			"text_type": embeddingType,
		},
	}

	// Send the request
	embeddings, err := client.sendRequest(body)
	if err != nil {
		return nil, err
	}
	defer embeddings.Body.Close()

	// Read the response body
	respBody, err := client.readResponseBody(embeddings)
	if err != nil {
		return nil, err
	}

	// Parse the response
	var response EmbeddingResponse
	if err := json.Unmarshal(respBody, &response); err != nil {
		return nil, fmt.Errorf("failed to parse the response: %w, response: %s", err, string(respBody))
	}

	// Check if there are embeddings in the response
	if len(response.Output.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings found in the response: %s", string(respBody))
	}

	// Collect all embeddings
	embeddingVectors := make([][]float64, len(response.Output.Embeddings))
	for i, embedding := range response.Output.Embeddings {
		embeddingVectors[i] = embedding.Embedding
	}

	return embeddingVectors, nil
}
