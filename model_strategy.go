package llm

import (
	"context"
	"fmt"
)

type ChatStrategy interface {
	Chat(ctx context.Context, chatMessages []ChatMessage, options *ChatOptions) (ChatResponse, error)
}

type EmbedStrategy interface {
	Embed(ctx context.Context, texts []string, options *EmbedOptions) (EmbedResponse, error)
}

// ModelContext supports separate strategies for chat and embed
type ModelContext struct {
	chatStrategy  ChatStrategy
	embedStrategy EmbedStrategy
}

func NewModelContext() *ModelContext {
	return &ModelContext{}
}

func (c *ModelContext) SetChatStrategy(strategy ChatStrategy) {
	c.chatStrategy = strategy
}

func (c *ModelContext) SetEmbedStrategy(strategy EmbedStrategy) {
	c.embedStrategy = strategy
}

func (c *ModelContext) Chat(ctx context.Context, chatMessages []ChatMessage, opts ...ChatOption) (ChatResponse, error) {
	if c.chatStrategy == nil {
		return nil, fmt.Errorf("chat strategy not set")
	}
	options := &ChatOptions{}
	for _, opt := range opts {
		opt(options)
	}
	return c.chatStrategy.Chat(ctx, chatMessages, options)
}

func (c *ModelContext) Embed(ctx context.Context, texts []string, opts ...EmbedOption) (EmbedResponse, error) {
	if c.embedStrategy == nil {
		return nil, fmt.Errorf("embedding strategy not set")
	}
	options := &EmbedOptions{}
	for _, opt := range opts {
		opt(options)
	}
	return c.embedStrategy.Embed(ctx, texts, options)
}
