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
  const thinkingLevel = userSettings.thinkingLevel || "medium";

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
  const maxInlineBase64Chars = 18_000_000;
  const maxBase64Chars = Math.floor(
    maxInlineBase64Chars / Math.max(1, attachedImages.length),
  );
  let inlineBase64Chars = 0;
  for (const imageUrl of attachedImages) {
    const image = await loadImageForGemini(imageUrl, maxBase64Chars);
    inlineBase64Chars += image.data.length;
    input.push(image);
  }
  if (inlineBase64Chars > maxInlineBase64Chars) {
    throw new Error(
      "The selected images are too large for Gemini's inline request limit.",
    );
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
        generation_config: {
          thinking_level: thinkingLevel,
        },
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

async function loadImageForGemini(url, maxBase64Chars) {
  const maxImageEdge = 4096;
  const supportedImageTypes = [
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
  ];
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load image: ${response.status}`);
  }

  const blob = await response.blob();
  const mimeType = blob.type.toLowerCase();
  if (!supportedImageTypes.includes(mimeType)) {
    throw new Error(`Unsupported image type: ${mimeType}`);
  }

  if (mimeType === "image/heic" || mimeType === "image/heif") {
    return inlineImage(mimeType, await blobToBase64(blob), maxBase64Chars);
  }

  const image = new Image();
  const canvas = document.createElement("canvas");
  const imageUrl = URL.createObjectURL(blob);
  try {
    image.src = imageUrl;
    await image.decode();

    const initialScale = Math.min(
      1,
      maxImageEdge / Math.max(image.naturalWidth, image.naturalHeight),
    );
    canvas.width = Math.max(1, Math.floor(image.naturalWidth * initialScale));
    canvas.height = Math.max(1, Math.floor(image.naturalHeight * initialScale));

    const context = canvas.getContext("2d", {
      colorSpace: "srgb",
      colorType: "unorm8",
    });
    if (!context) {
      throw new Error("Unable to prepare image: canvas is unavailable.");
    }

    const outputType = mimeType === "image/jpeg" ? "image/jpeg" : "image/png";
    while (true) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.imageSmoothingQuality = "high";
      context.drawImage(image, 0, 0, canvas.width, canvas.height);

      const encoded = await new Promise((resolve) =>
        canvas.toBlob(resolve, outputType, 0.92),
      );
      if (!encoded) {
        throw new Error("Unable to encode image.");
      }

      const data = await blobToBase64(encoded);
      if (data.length <= maxBase64Chars) {
        return inlineImage(outputType, data, maxBase64Chars);
      }
      if (canvas.width === 1 && canvas.height === 1) {
        throw new Error(
          "Image is too large for Gemini's inline request limit.",
        );
      }

      const scale = Math.min(0.8, Math.sqrt(maxBase64Chars / data.length) * 0.9);
      canvas.width = Math.max(1, Math.floor(canvas.width * scale));
      canvas.height = Math.max(1, Math.floor(canvas.height * scale));
    }
  } finally {
    canvas.width = canvas.height = 0;
    image.removeAttribute("src");
    URL.revokeObjectURL(imageUrl);
  }
}

function inlineImage(mimeType, data, maxBase64Chars) {
  if (data.length > maxBase64Chars) {
    throw new Error("Image is too large for Gemini's inline request limit.");
  }

  return {
    type: "image",
    mime_type: mimeType,
    data,
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
