package llm

type ChatOption func(option *ChatOptions)
type EmbedOption func(options *EmbedOptions)

func WithChatModel(model string) ChatOption {
	return func(c *ChatOptions) {
		c.Model = model
	}
}

func WithTemperature(temp float64) ChatOption {
	return func(c *ChatOptions) {
		c.Temperature = &temp
	}
}

func WithMaxTokens(tokens int) ChatOption {
	return func(c *ChatOptions) {
		c.MaxTokens = &tokens
	}
}

func WithTopP(topP float64) ChatOption {
	return func(c *ChatOptions) {
		c.TopP = &topP
	}
}

func WithStop(stop []string) ChatOption {
	return func(c *ChatOptions) {
		c.Stop = stop
	}
}

func WithEmbedModel(model string) EmbedOption {
	return func(e *EmbedOptions) {
		e.Model = model
	}
}

func WithEmbeddingType(textType string) EmbedOption {
	return func(e *EmbedOptions) {
		e.EmbeddingType = textType
	}
}

// TODO: add more options
