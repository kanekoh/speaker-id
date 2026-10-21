// player.js
// 識別ログの録音の再生（認証ヘッダーが要るので fetch して Blob で鳴らす）。同時に鳴るのは 1 つだけ
window.AudioPlayer = (function () {
  let player = null;
  let playing = null;  // { key, button }

  function stop() {
    if (player) { player.pause(); URL.revokeObjectURL(player.src); }
    if (playing && playing.button) playing.button.textContent = playing.button.dataset.label;
    player = null;
    playing = null;
  }

  async function toggle(entryId, kind, button) {
    const key = `${entryId}-${kind}`;
    const same = playing && playing.key === key;
    stop();
    if (same) return;  // 再生中のボタンをもう一度押したら停止
    playing = { key, button };
    button.textContent = "■ 停止";
    const res = await Auth.authFetch(`/identify_log/${entryId}/audio?kind=${kind}`);
    if (!res.ok) { stop(); alert(`再生できません (HTTP ${res.status})`); return; }
    if (!playing || playing.key !== key) return;  // 読み込み中に別のボタンが押された
    player = new Audio(URL.createObjectURL(await res.blob()));
    player.onended = stop;
    player.play();
  }

  // 再生ボタンを作る。画面を作り直しても、再生中ならその表示を引き継ぐ
  function button(entryId, kind, label, onClick) {
    const b = document.createElement("button");
    b.className = "btn btn-small";
    b.dataset.label = label;
    const key = `${entryId}-${kind}`;
    if (playing && playing.key === key) {
      playing.button = b;
      b.textContent = "■ 停止";
    } else {
      b.textContent = label;
    }
    b.onclick = () => { toggle(entryId, kind, b); if (onClick) onClick(); };
    return b;
  }

  return { button, stop };
})();
