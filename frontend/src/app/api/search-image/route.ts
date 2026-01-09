import { searchImage } from "@/lib/searchUtils";
import { NextResponse } from "next/server";

const imageCache = new Map<string, string>();

export const POST = async (req: Request) => {
  const body = await req.json();
  const query = body.query || "Eiffel Tower?";

  return handleSearch(query);
};

export const GET = async (req: Request) => {
  const { searchParams } = new URL(req.url);
  const query = searchParams.get("q") || searchParams.get("query");

  if (!query) {
    return NextResponse.json({ error: "Query parameter required" }, { status: 400 });
  }

  return handleSearch(query);
};

async function handleSearch(query: string) {
  const exists = imageCache.get(query);

  if (exists)
    return NextResponse.json({
      imageUrl: `/api/image-proxy?url=${encodeURIComponent(exists)}`,
    });

  try {
    const image = await searchImage(query);
    imageCache.set(query, image);

    const proxiedUrl = `/api/image-proxy?url=${encodeURIComponent(image)}`;
    return NextResponse.json({ imageUrl: proxiedUrl });
  } catch (error) {
    console.error(`Error searching image for "${query}":`, error);
    return NextResponse.json({ error: "Image search failed" }, { status: 500 });
  }
}
