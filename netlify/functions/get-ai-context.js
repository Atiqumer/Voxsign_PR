// netlify/functions/get-ai-context.js
const fetch = require('node-fetch');

exports.handler = async (event, context) => {
  // Only allow POST requests
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method Not Allowed" };
  }

  const { wordBuffer } = JSON.parse(event.body);
  const OPENROUTER_KEY = process.env.OPENROUTER_KEY; // Hides your key!

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENROUTER_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        "model": "google/gemma-2-9b-it:free",
        "messages": [
          { "role": "system", "content": "You are an ASL interpreter. Convert these letters into a short sentence. Return ONLY the sentence." },
          { "role": "user", "content": `Fragment: "${wordBuffer}"` }
        ]
      })
    });

    const data = await response.json();
    const aiSentence = data.choices[0].message.content;

    return {
      statusCode: 200,
      body: JSON.stringify({ sentence: aiSentence })
    };
  } catch (error) {
    return { statusCode: 500, body: JSON.stringify({ error: "Failed to connect to AI" }) };
  }
};