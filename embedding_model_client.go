package llm

// EmbeddingClient is an interface for interacting with embedding models.
type EmbeddingClient interface {
	GetEmbeddings(inputs []string, embeddingType string) ([][]float64, error)
}

// BaseEmbeddingClient is a base embedding client for interacting with embedding models.
type BaseEmbeddingClient struct {
	BaseModelClient
}
