"""Reference clues to identify page features from Headers, Cookies and Javascript Libraries"""
# features include elements of Security, Optimization, Technology and Services
# if total clues count > 0 -> feature considered present

LIST_HCJ_FEATURES = [{
    # SECURITY
    "name": "xss",
    "type": "secu",
    "CK": [],
    "HD": ["xss", "x-xss-protection", ],
    "JS": [],
}, {
    "name": "xsrf",
    "type": "secu",
    "CK": ["xsrf-token", "_csrf", "csrf", "csrftoken"],
    "HD": [],
    "JS": [],
}, {
    "name": "tls",
    "type": "secu",
    "CK": [],
    "HD": ["strict-transport-security"],
    "JS": [],
}, {
    "name": "contentPolicy",
    "type": "secu",
    "CK": [],
    "HD": ["x-frame-options", "content-security-policy", "x-permitted-cross-domain-policies",
           "access-control-allow-origin", "cross-origin-opener-policy", "cross-origin-resource-policy",
           "cross-origin-embedder-policy", "access-control-allow-methods", "content-security-policy-report-only", ],
    "JS": [],
}, {
    "name": "captcha",
    "type": "secu",
    "CK": ["captcha-tracker"],
    "HD": [],
    "JS": [],
},

    # OPTIMIZATION
    {
        "name": "cache",
        "type": "opti",
        "CK": ["ssr-caching"],
        "HD": ["cache-control", "cf-cache-status", "x-cache", "pragma", "x-proxy-cache", "x-cache-miss-from",
               "x-cacheable",
               "x-litespeed-cache", "x-cache-group", "server-cache", "x-cache-enabled", "x-cache-hit",
               "x-cacheproxy-retries",
               "x-cache-status", "x-proxy-cache-info", "x-sucuri-cache"],
        "JS": [],
    }, {
        "name": "etag",
        "type": "opti",
        "CK": [],
        "HD": ["etag"],
        "JS": [],
    }, {
        "name": "parkingPage",
        "type": "opti",
        "CK": ["parking_session"],
        "HD": [],
        "JS": [],
    },

    # TECHNOLOGY
    {
        "name": "ASP",
        "type": "techno",
        "CK": ["asp.net_sessionid"],
        "HD": ["x-aspnet-version", "x-aspnetmvc-version"],
        "JS": [],
    }, {
        "name": "PHP",
        "type": "techno",
        "CK": ["phpsessid"],
        "HD": ["x-php-version"],
        "JS": [],
    }, {
        "name": "AWS",
        "type": "techno",
        "CK": ["awsalb", "awsalbtg"],
        "HD": ["x-amz-cf-id", "x-amz-cf-pop", "x-amz-request-id", "x-amz-id-2", "x-amz-server-side-encryption"],
        "JS": [],
    }, {
        "name": "Java",
        "type": "techno",
        "CK": ["jsessionid"],
        "HD": [],
        "JS": [],
    },
    {
        "name": "Jquery",
        "type": "techno",
        "CK": [],
        "HD": [],
        "JS": ["jquery.js", "jquery-migrate.js", "slick.js", "jquery.blockui.js", "jquery-ui.js",
               "jquery.waypoints.js"],
    },
    {
        "name": "React",
        "type": "techno",
        "CK": [],
        "HD": [],
        "JS": ["react-dom.production.js", "react.production.js", "fusion-alert.js", "fusion-flexslider.js",
               "fusion-general-global.js", "fusion-lightbox.js", "fusion-video-general.js", "fusion-column.js",
               "fusion-container.js", "fusion-parallax.js"],
    },
    {
        "name": "Angular",
        "type": "techno",
        "CK": [],
        "HD": [],
        "JS": ["angular.js", "angular-translate.js", "angular-locale_en.js"],
    },

    # SERVICES
    {
        "name": "adblock",
        "type": "service",
        "CK": [],
        "HD": ["x-adblock-key"],
        "JS": [],
    }, {
        "name": "wix",
        "type": "service",
        "CK": [],
        "HD": ["x-wix-request-id", ],
        "JS": ["wix-perf-measure.umd.js"],
    }, {
        "name": "wix",
        "type": "service",
        "CK": ["wordpress_test_cookie"],
        "HD": [],
        "JS": [],
    }, {
        "name": "shopify",
        "type": "service",
        "CK": ["_shopify_s", "_shopify_y", "_shopify_m", "_shopify_tm", "_shopify_tw"],
        "HD": ["x-shopify-stage", ],
        "JS": [],
    }, {
        "name": "shop",
        "type": "service",
        "CK": ["shop_session_token"],
        "HD": ["x-shopify-stage", "x-storefront-renderer-rendered", "x-shopid", "x-sorting-hat-shopid"],
        "JS": ["woocommerce.js", "woocommerce-add-to-cart.js", ],
    }, {
        "name": "sucuri",
        "type": "service",
        "CK": [],
        "HD": ["x-sucuri-id", "x-sucuri-cache", ],
        "JS": [],
    }, {
        "name": "googleAds",
        "type": "service",
        "CK": [],
        "HD": [],
        "JS": ["adsbygoogle.js"],
    },
]

all_js = []
all_hds = []
all_cks = []
for e in LIST_HCJ_FEATURES:
    all_js += e["JS"]
    all_hds += e["HD"]
    all_cks += e["CK"]
SET_ALL_CONSIDERED_JS = set(all_js)
SET_ALL_CONSIDERED_HDS = set(all_hds)
SET_ALL_CONSIDERED_CKS = set(all_cks)
