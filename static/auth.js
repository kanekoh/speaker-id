// auth.js
window.Auth = (function () {
  let apiToken = "";

  function logout() {
    localStorage.removeItem("api_token");
    apiToken = "";
    // 認証ダイアログを再表示
    showTokenDialog();
  }

  // トークン入力ダイアログ作成・表示
  function showTokenDialog(onSuccess) {
    let dlg = document.createElement("div");
    dlg.style = "background:#fff;border:1px solid #ccc;padding:2em;position:fixed;top:20vh;left:50%;transform:translateX(-50%);z-index:10";
    dlg.innerHTML = `
      <label>APIトークンを入力してください</label>
      <input type="password" id="tokenInput" autocomplete="off">
      <button id="tokenBtn">認証して開始</button>
      <p id="tokenStatus" style="color:red"></p>
    `;
    document.body.appendChild(dlg);

    document.getElementById("tokenBtn").onclick = async function () {
      const token = document.getElementById("tokenInput").value.trim();
      if (!token) {
        document.getElementById("tokenStatus").textContent = "トークンを入力してください";
        return;
      }
      // /check_token でバリデーション
      try {
        const res = await fetch("/check_token", {
          headers: { "Authorization": "Bearer " + token }
        });
        if (res.ok) {
          localStorage.setItem("api_token", token);
          apiToken = token;
          dlg.remove();
          if (onSuccess) onSuccess(token);
        } else {
          document.getElementById("tokenStatus").textContent = "認証失敗。正しいトークンを入力してください";
        }
      } catch {
        document.getElementById("tokenStatus").textContent = "通信エラー";
      }
    };
  }

  async function ensureToken(onSuccess) {
    const token = localStorage.getItem("api_token");
    if (token) {
      // 既存トークンでバリデーション
      const res = await fetch("/check_token", {
        headers: { "Authorization": "Bearer " + token }
      });
      if (res.ok) {
        apiToken = token;
        if (onSuccess) onSuccess(token);
        return;
      }
    }
    // 未保存 or 不正ならダイアログ表示
    showTokenDialog(onSuccess);
  }

  // fetchラッパー（認証ヘッダーを自動で付与）
  async function authFetch(url, opts = {}) {
    if (!apiToken) {
      apiToken = localStorage.getItem("api_token") || "";
    }
    opts.headers = opts.headers || {};
    opts.headers["Authorization"] = "Bearer " + apiToken;
    return fetch(url, opts);
  }

  return {
    ensureToken,
    authFetch,
    logout
  };
})();
