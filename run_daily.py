def generate_mcqs(news_list):
    log(f"Step 3: Generating {len(news_list)} exam-oriented factual MCQs...")
    
    prompt = f"""
    You are an expert content developer for UPSC/UPPCS. 
    Create 10 Factual MCQs that connect Current News with Static General Studies.
    
    RULES FOR EXAM-STANDARD QUESTIONS:
    1. THE 'BACKWARD LINKAGE' RULE: Don't just ask about the headline. Ask about the associated Ministry, the Act/Law, the Location, or the parent Organization.
    2. FACTUAL PRECISION: Use exact data, percentages, or constitutional articles mentioned or implied.
    3. NO OBVIOUS ANSWERS: Options should be closely related.
    
    TASK EXAMPLES:
    - If news is about 'Trump/Oil', ask about 'India's top oil supplier' or 'Percentage of import'.
    - If news is about 'Coast Guard', ask about its 'Statutory status' or 'Nodal Ministry'.
    - If news is about 'Pongal', ask about the 'Sangam literature' reference or 'Tamil Calendar month'.
    
    NEWS DATA:
    {json.dumps(news_list)}

    OUTPUT FORMAT (Strict JSON):
    {{
      "mcqs": [
        {{
          "question": "Example: The 'Surya' class pollution control vessels, recently in news, are being inducted into which of the following organizations?",
          "options": ["Indian Navy", "Indian Coast Guard", "Marine Police", "Directorate General of Shipping"],
          "correct_index": 1,
          "source": "..."
        }}
      ]
    }}
    """

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional examiner focusing on factual 'Static-Linkage' questions. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2 
        )
        
        raw_res = json.loads(resp.choices[0].message.content)
        mcqs = raw_res.get("mcqs") or list(raw_res.values())[0]
        log(f"Successfully generated {len(mcqs)} exam-standard factual MCQs.")
        return mcqs
    except Exception as e:
        log(f"AI ERROR: {e}")
        return []
