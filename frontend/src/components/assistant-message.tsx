import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { ChatMessage } from "../../generated";
import { ImageSection, ImageSectionSkeleton } from "./image-section";
import { MessageComponent, MessageComponentSkeleton } from "./message";
import RelatedQuestions from "./related-questions";
import { SearchResults, SearchResultsSkeleton } from "./search-results";
import { Section } from "./section";

export function ErrorMessage({ content }: { content: string }) {
  return (
    <Alert className="bg-red-500/5 border-red-500/15 p-5">
      <AlertCircle className="h-4 w-4 stroke-red-500 stroke-2" />
      <AlertDescription className="text-base text-foreground">
        {content.split(" ").map((word, index) => {
          const urlPattern = /(https?:\/\/[^\s]+)/g;
          if (urlPattern.test(word)) {
            return (
              <a
                key={index}
                href={word}
                target="_blank"
                rel="noopener noreferrer"
                className="underline"
              >
                {word}
              </a>
            );
          }
          return word + " ";
        })}
      </AlertDescription>
    </Alert>
  );
}

export const AssistantMessageContent = ({
  message,
  isStreaming = false,
  onRelatedQuestionSelect,
}: {
  message: ChatMessage;
  isStreaming?: boolean;
  onRelatedQuestionSelect: (question: string) => void;
}) => {
  const {
    sources,
    content,
    related_queries,
    images,
    is_error_message = false,
  } = message;

  if (is_error_message) {
    return <ErrorMessage content={message.content} />;
  }

  // Parse Expert Plan (DeepResearch)
  let expertPlan = "";
  let finalContent = content;
  
  const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
  if (thinkMatch) {
      expertPlan = thinkMatch[1].trim();
      finalContent = content.replace(/<think>[\s\S]*?<\/think>/, "").trim();
  } else if (content.includes("<think>")) {
      // Incomplete streaming detected
      const parts = content.split("<think>");
      if (parts.length > 1) {
          expertPlan = parts[1].trim();
          finalContent = parts[0].trim(); // Usually empty if think is at start
      }
  }

  return (
    <div className="flex flex-col">
       {expertPlan && (
        <Section title="Expert Plan" animate={isStreaming}>
          <div className="prose prose-sm dark:prose-invert bg-card p-4 rounded-md border border-border/50 shadow-sm whitespace-pre-wrap">
            {expertPlan}
          </div>
        </Section>
      )}
      <Section title="Answer" animate={isStreaming} streaming={isStreaming}>
        {content ? (
          <MessageComponent message={{...message, content: finalContent}} isStreaming={isStreaming} />
        ) : (
          <MessageComponentSkeleton />
        )}
      </Section>
      <Section title="Sources" animate={isStreaming}>
        {!sources || sources.length === 0 ? (
          <SearchResultsSkeleton />
        ) : (
          <>
            <SearchResults results={sources} />
          </>
        )}
      </Section>
      <Section title="Images" animate={isStreaming}>
        {images && images.length > 0 ? (
          <ImageSection images={images} />
        ) : (
          <ImageSectionSkeleton />
        )}
      </Section>
      {related_queries && related_queries.length > 0 && (
        <Section title="Related" animate={isStreaming}>
          <RelatedQuestions
            questions={related_queries}
            onSelect={onRelatedQuestionSelect}
          />
        </Section>
      )}
    </div>
  );
};
