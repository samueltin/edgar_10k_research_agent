workspace "10-K Research Agent" "C4 model for an AI-assisted equity research prototype (SEC EDGAR, single-company workflow)" {

    model {
        analyst = person "Research analyst" "Builds a research view or model on a single company at a time."

        researchAgentSystem = softwareSystem "10-K Research Agent" "Extracts structured KPIs and segment data from a company's 10-K filing, validates them, and presents a research memo with follow-up chat." {

            streamlitUi = container "Streamlit UI" "Collects the ticker, renders the research memo, and hosts the follow-up chat." "Python, Streamlit"

            researchAgent = container "Research Agent" "Orchestrates extraction and validation of KPIs and segment data from a 10-K filing." "Python, LangGraph" {

                xbrlExtractor = component "XBRL extractor" "Pulls structured financial facts (revenue, gross profit, net income) from XBRL company facts. Deterministic, no LLM." "Python, edgartools"

                sectionExtractor = component "Section extractor" "Pulls MD&A (Item 7) and Risk Factors (Item 1A) text from the latest 10-K." "Python, edgartools"

                segmentExtractor = component "Segment extractor" "Extracts segment-level revenue from MD&A/footnote text using LLM structured output." "Python, LangGraph node"

                validator = component "Validator" "Compares the sum of LLM-extracted segment revenue against the XBRL consolidated total; flags mismatches beyond threshold." "Python, LangGraph node"

                humanReviewFlag = component "Human review flag" "Marks a filing's segment data as flagged for manual review when validation fails." "Python, LangGraph node"
            }

            edgarClient = container "EDGAR Client" "Thin wrapper around SEC EDGAR access (filings, XBRL facts, section text). Candidate for a future MCP server." "Python, edgartools"
        }

        azureOpenAI = softwareSystem "Azure OpenAI" "Hosts the LLM used for segment extraction and the analyst chat." "External System"
        secEdgar = softwareSystem "SEC EDGAR" "Public SEC filing and XBRL data source." "External System"

        analyst -> streamlitUi "Views memo, asks follow-up questions"

        streamlitUi -> researchAgent "Invokes for a given ticker"
        researchAgent -> edgarClient "Requests filing text and XBRL facts"

        sectionExtractor -> segmentExtractor "Passes MD&A / footnote text"
        xbrlExtractor -> validator "Provides ground-truth total revenue"
        segmentExtractor -> validator "Provides extracted segment revenue"
        validator -> humanReviewFlag "Routes on validation failure"
        validator -> streamlitUi "Routes validated data to memo (on pass)"

        edgarClient -> secEdgar "Fetches filings, XBRL facts" "HTTPS/JSON"
        segmentExtractor -> azureOpenAI "Structured extraction call" "HTTPS/JSON"
        streamlitUi -> azureOpenAI "Chat completion calls" "HTTPS/JSON"
    }

    views {
        systemContext researchAgentSystem "SystemContext" {
            include *
            autoLayout
            description "Context: the research analyst uses the 10-K Research Agent, which depends on Azure OpenAI and SEC EDGAR."
        }

        container researchAgentSystem "Containers" {
            include *
            autoLayout
            description "Container view: Streamlit UI, Research Agent, and EDGAR Client, plus external dependencies."
        }

        component researchAgent "Components" {
            include *
            autoLayout
            description "Component view: extraction and validation pipeline inside the Research Agent."
        }

        styles {
            element "Person" {
                shape person
                background #999999
                color #ffffff
            }
            element "Software System" {
                background #1168bd
                color #ffffff
            }
            element "External System" {
                background #999999
                color #ffffff
            }
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Component" {
                background #85bbf0
                color #000000
            }
        }
    }
}
