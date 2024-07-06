package llm

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatOptions struct {
	Model       string   `json:"model"`
	Temperature *float64 `json:"temperature,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	// TODO: add more options
}

type EmbedOptions struct {
	Model         string `json:"model"`
	EmbeddingType string `json:"embedding_type,omitempty"`
	// TODO: add more options
}

type ChatResponse interface {
	GetContent() string
}

type EmbedResponse interface {
	GetEmbeddings() [][]float32
}
