import {defineConfig} from 'vitepress'

export default defineConfig({
    lang: 'zh',
    title: "AI & ML",
    description: "AI & ML",
    head: [
        ['link', {rel: 'icon', href: '/favicon-32x32.png'}]
    ],
    lastUpdated: true,
    themeConfig: {
        logo: '/favicon.ico',
        outline: {
            level: [2, 3]
        },
        nav: [
            {text: '主页', link: '/'},
            {text: '基本概念', link: '/concepts/supervised-learning'},
            {text: '常用算法', link: '/algorithm/linear-regression'},
            {text: '模型评估', link: '/assessment/accuracy'},
        ],
        sidebar: {
            '/concepts/': [
                {
                    text: '基本概念',
                    items: [
                        {text: '基本概念 监督学习', link: '/concepts/supervised-learning'},
                        {text: '基本概念 无监督学习', link: '/concepts/unsupervised-learning'},
                        {text: '基本概念 强化学习', link: '/concepts/reinforcement-learning'},
                        {text: '基本概念 过拟合', link: '/concepts/overfitting'},
                        {text: '基本概念 欠拟合', link: '/concepts/underfitting'},
                        {text: '基本概念 偏差', link: '/concepts/deviation'},
                        {text: '基本概念 方差', link: '/concepts/variance'},
                    ]
                }
            ],
            '/algorithm/': [
                {
                    text: '常用算法',
                    items: [
                        {text: '常用算法 线性回归', link: '/algorithm/linear-regression'},
                        {text: '常用算法 逻辑回归', link: '/algorithm/logistic-regression'},
                        {text: '常用算法 支持向量机', link: '/algorithm/support-vector-machine'},
                        {text: '常用算法 决策树', link: '/algorithm/decision-tree'},
                        {text: '常用算法 随机森林', link: '/algorithm/random-forest'},
                        {text: '常用算法 K 近邻', link: '/algorithm/knn'},
                        {text: '常用算法 K-Means', link: '/algorithm/k-means'},
                    ]
                }
            ],
            '/assessment/': [
                {
                    text: '模型评估',
                    items: [
                        {text: '模型评估 准确率', link: '/assessment/accuracy'},
                        {text: '模型评估 精确率', link: '/assessment/accuracy-rate'},
                        {text: '模型评估 召回率', link: '/assessment/recall-rate'},
                        {text: '模型评估 F1 值', link: '/assessment/f1-value'},
                        {text: '模型评估 ROC 曲线', link: '/assessment/roc-curve'},
                        {text: '模型评估 AUC', link: '/assessment/auc'},
                    ]
                }
            ]
        },
        socialLinks: [
            {icon: 'github', link: 'https://github.com/xuehang-org/ai-ml'},
            {icon: 'twitter', link: 'https://x.com/xuehang_org'},
        ],
        footer: {
            message: '基于 MIT 许可发布',
            copyright: '版权所有 © 2025 xuehang.org'
        },
        search: {
            provider: 'local'
        },
    },
    markdown: {
        math: true,
        container: {
            tipLabel: '提示',
            warningLabel: '警告',
            dangerLabel: '危险',
            infoLabel: '信息',
            detailsLabel: '详细信息',
        }
    }
})