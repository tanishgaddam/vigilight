/* VigiLight add-on: demo video dropdown + hides Telegram / Discord from the page.
   Add this line just before </body> in static/index.html:
       <script src="/static/demo-picker.js"></script>                                   */
(function () {
  "use strict";

  // ---- 1. demo video picker ------------------------------------------------------
  var css = document.createElement("style");
  css.textContent =
    "#vl-demo{position:fixed;right:14px;bottom:14px;z-index:9000;background:#12161c;color:#e8eef5;" +
    "border:1px solid #2c3540;border-radius:10px;padding:10px 12px;font:13px system-ui,sans-serif;" +
    "box-shadow:0 6px 24px rgba(0,0,0,.45);min-width:250px}" +
    "#vl-demo b{display:block;margin-bottom:6px;font-size:12px;letter-spacing:.04em;color:#8fa3b8;text-transform:uppercase}" +
    "#vl-demo select{width:100%;padding:7px;border-radius:6px;background:#0b0e12;color:inherit;border:1px solid #2c3540}" +
    "#vl-demo .row{display:flex;gap:6px;margin-top:6px}" +
    "#vl-demo button{flex:1;padding:6px;border-radius:6px;border:1px solid #2c3540;background:#1b222b;color:inherit;cursor:pointer}" +
    "#vl-demo button:hover{background:#26303c}" +
    "#vl-demo small{display:block;margin-top:6px;color:#8fa3b8;min-height:15px}";
  document.head.appendChild(css);

  var box = document.createElement("div");
  box.id = "vl-demo";
  box.innerHTML =
    "<b>Demo video</b><select id='vl-demo-sel'><option value=''>Loading videos...</option></select>" +
    "<div class='row'><button id='vl-demo-play'>Play</button><button id='vl-demo-stop'>Stop demo</button>" +
    "<button id='vl-demo-ref' title='Rescan videos folder'>&#8635;</button></div><small id='vl-demo-msg'></small>";
  document.body.appendChild(box);

  var sel = box.querySelector("#vl-demo-sel"), msg = box.querySelector("#vl-demo-msg");
  function say(t) { msg.textContent = t || ""; }

  function post(url, body) {
    return fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) })
      .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, data: j }; }); });
  }

  function loadVideos() {
    fetch("/api/demo/videos").then(function (r) { return r.json(); }).then(function (list) {
      sel.innerHTML = "";
      if (!list.length) {
        sel.innerHTML = "<option value=''>No videos found (add to videos/)</option>";
        return;
      }
      var ph = document.createElement("option");
      ph.value = ""; ph.textContent = "-- choose a video --";
      sel.appendChild(ph);
      list.forEach(function (v) {
        var o = document.createElement("option");
        o.value = v.id; o.textContent = (v.playing ? "\u25B6 " : "") + v.label;
        if (v.playing) o.selected = true;
        sel.appendChild(o);
      });
    }).catch(function () { say("Could not load the video list."); });
  }

  function play() {
    if (!sel.value) { say("Choose a video first."); return; }
    say("Loading models, the video starts in a few seconds...");
    post("/api/demo/start", { video: sel.value }).then(function (res) {
      say(res.ok ? "Playing: " + sel.options[sel.selectedIndex].text.replace("\u25B6 ", "") : (res.data.error || "Failed"));
      loadVideos();
    });
  }

  sel.addEventListener("change", play);                      // picking a video starts it
  box.querySelector("#vl-demo-play").addEventListener("click", play);
  box.querySelector("#vl-demo-ref").addEventListener("click", loadVideos);
  box.querySelector("#vl-demo-stop").addEventListener("click", function () {
    post("/api/demo/stop").then(function () { say("Demo stopped."); loadVideos(); });
  });
  loadVideos();

  // ---- 2. hide Telegram / Discord UI ----------------------------------------------
  var RX = /telegram|discord/i;
  function hide() {
    document.querySelectorAll("button,a,li,label,span,div,.chip,.badge,.pill,.tag").forEach(function (el) {
      if (el.closest("#vl-demo") || el.dataset.vlHidden) return;
      var txt = (el.textContent || "").trim();
      if (!txt || txt.length > 40 || !RX.test(txt) || el.querySelectorAll("button,a,input,select").length > 1) return;
      var target = el.closest("button,.chip,.badge,.pill,.tag,li,label") || el;   // hide the whole chip/button
      target.style.display = "none";
      target.dataset.vlHidden = "1";
    });
  }
  hide();
  new MutationObserver(hide).observe(document.body, { childList: true, subtree: true });   // channel chips are re-rendered by the page
})();
