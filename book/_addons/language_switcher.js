// book/_addons/language_switcher.js
function switchLanguage(newLang) {
    var currentPath = window.location.pathname;
    var supportedLangs = __SUPPORTED_LANGUAGES__; // This will be replaced by the actual languages during preprocessing
    var langRegex = new RegExp('\\b(' + supportedLangs.join('|') + ')\\b');

    if (langRegex.test(currentPath)) {
        // If the current path contains a language code, replace it
        var newPath = window.location.origin + currentPath.replace(langRegex, newLang);
        window.location.href = newPath + window.location.search + window.location.hash;
    }
}
