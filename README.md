# mash

MATLAB agent shell. A tui to work with Claude Code (can use subscription) in persistent matlab sessions.

<table width="100%">
  <tr>
    <th>Example</th>
  </tr>
  <tr>
    <td width="100%">
      <img src="./docs/screenshot.png" width="100%" />
    </td>
  </tr>
</table>

Built with
- jupyter-matlab-proxy
- agent-client-protocol

Capabilities:
- sessions and workspaces
- render images (if terminal supports)
- save session scripts as .ipynb and .m

Install:
```
bun install
uv run mash-setup
uv run mash
# Don't run uv sync as it will remove the matlabengine for python (need to rerun uv run mash-setup if sync)
```
