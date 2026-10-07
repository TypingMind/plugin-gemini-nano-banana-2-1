async function gemini_nano_banana_2_1(
  params,
  userSettings,
  authorizedResources,
) {
  const prompt = params.prompt;
  const geminikey = userSettings.geminikey;
  const model = "gemini-nano-banana-2.1";
  const aspectRatio = userSettings.aspectRatio || "auto";
  const imageSize = userSettings.imageSize || "auto";

  if (!geminikey) {
    throw new Error(
      "No Gemini API key provided. Please enter your Gemini API key in the plugin settings and try again.",
    );
  }

  let attachedImages;
  if (params.images === undefined) {
    const cards = authorizedResources?.previousRunOutput?.cards;
    attachedImages = (Array.isArray(cards) ? cards : [])
      .filter((card) => card.type === "image" && card.image?.url)
      .map((card) => card.image.url);
  } else {
    if (!Array.isArray(params.images)) {
      throw new Error("images must be an array of attachment ids.");
    }

    attachedImages = params.images.map((id) => {
      const attachment = (authorizedResources?.attachments || []).find(
        (item) => item.id === id,
      );
      if (!attachment?.type?.startsWith("image/") || !attachment.url) {
        throw new Error(`Image attachment is not available: ${id}`);
      }
      return attachment.url;
    });
  }

  if (attachedImages.length > 14) {
    throw new Error("Gemini Nano Banana 2.1 supports up to 14 reference images.");
  }

  const input = [{ type: "text", text: prompt }];
  for (const imageUrl of attachedImages) {
    input.push(await loadImageForGemini(imageUrl));
  }

  const responseFormat = { type: "image" };
  if (aspectRatio !== "auto") responseFormat.aspect_ratio = aspectRatio;
  if (imageSize !== "auto") responseFormat.image_size = imageSize;

  const response = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/interactions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": geminikey,
      },
      body: JSON.stringify({
        model,
        input,
        response_format: responseFormat,
      }),
    },
  );

  if (response.status === 401 || response.status === 403) {
    throw new Error("Invalid Gemini API key or insufficient API permissions.");
  }
  if (!response.ok) {
    throw new Error(`Gemini API error (${response.status}): ${await response.text()}`);
  }

  const data = await response.json();
  const outputImage = getOutputImage(data);
  if (!outputImage?.data) {
    throw new Error("Gemini response did not contain an output image.");
  }

  return {
    cards: [
      {
        type: "image",
        image: {
          url: `data:${outputImage.mime_type || "image/png"};base64,${outputImage.data}`,
          alt: prompt.replace(/[\[\]]/g, ""),
          filename:
            typeof params.filename === "string" && params.filename.trim()
              ? params.filename.trim()
              : undefined,
        },
      },
    ],
  };
}

function getOutputImage(data) {
  if (data.output_image?.data) {
    return data.output_image;
  }

  for (const step of data.steps || []) {
    if (step.type !== "model_output") continue;
    for (const content of step.content || []) {
      if (content.type === "image" && content.data) return content;
    }
  }

  return null;
}

async function loadImageForGemini(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load image: ${response.status}`);
  }

  const blob = await response.blob();
  const mimeType = blob.type || "image/png";
  if (!mimeType.startsWith("image/")) {
    throw new Error(`Unsupported image type: ${mimeType}`);
  }

  return {
    type: "image",
    mime_type: mimeType,
    data: await blobToBase64(blob),
  };
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = () => reject(new Error("Unable to read image data."));
    reader.readAsDataURL(blob);
  });
}
