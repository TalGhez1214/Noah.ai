AGENT_PROMPT = """ ## Role ##
            You will receive an article dicitionary with title, auther, publish date, content etc. Extract the following:
            1. summary: A short 2–3 sentence summary.
            2. quote: The most relevant quote from the article to the user query (1-2 sentences with more then 5 words).

            user_query: {user_query}

            ## Rules ##
            - Default behavior: If the article has overlap with the user’s query (same topic or subtopic, contain main keywords, title etc), produce a summary.
            - Reject only if: The article’s topic is very different from the user query (no reasonable connection).
            - Quote rule (verbatim): it must be copied exactly from the article (punctuation/casing preserved). 
            - No inventions: Do not add facts, numbers, dates, or interpretations that aren’t explicitly in the article.

            ## Article ##

            {article}
        """