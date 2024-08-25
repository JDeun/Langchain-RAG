class NewsletterGenerator:
    def __init__(self, llm, prompts, config):
        self.llm = llm
        self.prompts = prompts
        self.config = config

    def generate_content(self, context, queries):
        newsletter = f"# {self.config['NEWSLETTER_TOPIC']}\n\n"
        newsletter += self._generate_hot_issues(context["hot_issues"])
        newsletter += self._generate_tech_snapshots(context["tech_snapshots"])
        newsletter += self._generate_ai_shorts(context["ai_shorts"])
        return newsletter

    def _generate_hot_issues(self, hot_issues):
        content = "## 이번 주의 AI 핫이슈\n\n"
        for issue in hot_issues[:2]:
            content += f"### {issue['query']}\n"
            content += f"{issue['result']}\n\n"
        return content

    def _generate_tech_snapshots(self, tech_snapshots):
        content = "## AI 기술 스냅샷\n\n"
        snapshots = tech_snapshots.split('\n\n')
        for snapshot in snapshots[:2]:
            content += f"{snapshot}\n\n"
        return content

    def _generate_ai_shorts(self, ai_shorts):
        content = "## AI 세계 단신\n\n"
        shorts = ai_shorts.split('\n')
        for short in shorts[:5]:
            if short.strip():
                content += f"- {short}\n"
        return content