// Talks only to the local Flask endpoint. No external calls.
const input    = document.getElementById("input");
const output   = document.getElementById("output");
const go        = document.getElementById("go");
const status    = document.getElementById("status");
const charcount = document.getElementById("charcount");
let mode = "beam";

input.addEventListener("input", () => {
  charcount.textContent = `${input.value.length} chars`;
});

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mode").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-checked", "false");
    });
    btn.classList.add("active");
    btn.setAttribute("aria-checked", "true");
    mode = btn.dataset.mode;
  });
});

async function translate() {
  const text = input.value.trim();
  if (!text) {
    input.focus();
    return;
  }
  go.disabled = true;
  status.textContent = "decoding…";
  output.classList.remove("error");

  try {
    const res = await fetch("/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");

    output.textContent = data.translation || "—";
    output.dataset.empty = "false";
    status.textContent = `${mode} · done`;
  } catch (err) {
    output.textContent = err.message;
    output.classList.add("error");
    output.dataset.empty = "false";
    status.textContent = "error";
  } finally {
    go.disabled = false;
  }
}

go.addEventListener("click", translate);

// Ctrl/Cmd + Enter to translate
input.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") translate();
});
