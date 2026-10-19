// auth.js
window.Auth = (function () {
  let apiToken = "";

  function logout() {
    localStorage.removeItem("api_token");
    apiToken = "";
    // 認証ダイアログを再表示
    showTokenDialog(() => location.reload());
  }

  // トークン入力ダイアログ作成・表示（見た目は style.css の .auth-*）
  function showTokenDialog(onSuccess) {
    const backdrop = document.createElement("div");
    backdrop.className = "auth-backdrop";
    backdrop.innerHTML = `
      <form class="card auth-dialog" role="dialog" aria-modal="true" aria-labelledby="tokenTitle">
        <h2 id="tokenTitle">APIトークンを入力してください</h2>
        <div class="field">
          <label for="tokenInput">APIトークン</label>
          <input type="password" id="tokenInput" autocomplete="off">
        </div>
        <div class="actions">
          <button type="submit" id="tokenBtn" class="btn btn-primary">認証して開始</button>
        </div>
        <p id="tokenStatus" class="auth-error" role="alert"></p>
      </form>
    `;
    document.body.appendChild(backdrop);
    const input = backdrop.querySelector("#tokenInput");
    const status = backdrop.querySelector("#tokenStatus");
    input.focus();

    backdrop.querySelector("form").onsubmit = async function (e) {
      e.preventDefault();
      const token = input.value.trim();
      if (!token) {
        status.textContent = "トークンを入力してください";
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
          backdrop.remove();
          if (onSuccess) onSuccess(token);
        } else {
          status.textContent = "認証失敗。正しいトークンを入力してください";
        }
      } catch {
        status.textContent = "通信エラー";
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
