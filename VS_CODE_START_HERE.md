# Your first steps in VS Code

VS Code is where you write and organize files. A programming language's runtime
or compiler runs your code. Extensions add features such as suggestions,
formatting, and debugging support.

## 1. Open your project folder

Choose **File > Open Folder** and select:

`C:\Users\Mohan\OneDrive\Desktop\Project_Fav`

Opening a folder gives VS Code a home for your files, searches, terminal, and
project settings. If asked about Workspace Trust, trust a folder only when you
know and trust its contents.

In the Explorer, open this file. Press **Ctrl+Shift+V** to see its formatted
Markdown preview. Markdown is plain text with simple formatting markers such
as `#` for headings.

## 2. Get to know the screen

| Area | What you use it for |
| --- | --- |
| Activity Bar, usually at the far left | Switch between Explorer, Search, Source Control, Run and Debug, and Extensions. |
| Explorer sidebar | Create, open, rename, and organize your project's files. |
| Editor in the center | Read and edit files; each open file has a tab. |
| Bottom panel | Use the terminal and inspect Problems, Output, or debugging information. |
| Status Bar along the bottom | Check the language mode, cursor position, and other project information. |
| Secondary sidebar, often on the right | Hosts additional views, commonly chat. |

**Command Palette:** press **Ctrl+Shift+P** and type what you want to do.
For example, search for `Preferences: Open Settings (UI)` or `Color Theme`.
This is a useful starting point whenever you cannot find a command.
## 3. Try a five-minute exercise


1. Press **Ctrl+Shift+E** to focus the Explorer.
2. Use its **New File** button to create `practice.txt` inside Project_Fav.
3. Type `Hello, Mohan! I am learning VS Code.` and press **Ctrl+S**.
4. Add two more lines. Put the cursor on one and press **Alt+Up** or
   **Alt+Down** to move it.
5. Press **Ctrl+F** and search for `learning` in this file.
6. Press **Ctrl+P**, type `VS_CODE_START_HERE`, and reopen this guide.
7. Choose **Terminal > New Terminal**. In PowerShell, enter `Get-Location`
   to see the current folder, then `Get-ChildItem` to list its contents.
8. Open `practice.txt` again and change a word. Press **Ctrl+Z** to undo it.

The terminal is a command interface to your computer. The editor changes file
contents; the terminal runs commands. Clicking an editor tab does not change
the terminal's current folder.

## 4. Learn these shortcuts first

These are the standard Windows bindings; custom keybindings can change them.

| Action | Shortcut |
| --- | --- |
| Find a command | Ctrl+Shift+P |
| Find and open a file | Ctrl+P |
| Save | Ctrl+S |
| Undo | Ctrl+Z |
| Search this file | Ctrl+F |
| Search the whole folder | Ctrl+Shift+F |
| Show or hide the primary sidebar | Ctrl+B |
| Open settings | Ctrl+, |
| Toggle the terminal | Ctrl+` (backtick) |
| Open Extensions | Ctrl+Shift+X |

## 5. Understand your setup

This folder's `.vscode/settings.json` sets:

- Editor text to 15 pixels and terminal text to 14 pixels.
- Line numbers on, so it is easy to locate a line.
- Word wrap on, so long lines fit the editor without changing the actual file.
- The minimap off, leaving more space for your text.
- Matching brackets in colors, making nested code easier to follow.
- Preview tabs off, so opening another file keeps the previous tab available.

Your existing user settings select Dark+ and enable Auto Save after a delay.
Ctrl+S is still useful when you want to save immediately.

**User settings** apply generally across projects. **Workspace settings** apply
to the opened project and override corresponding user settings. To adjust these
preferences, press **Ctrl+,** and choose the **Workspace** tab.

To reverse a preference added here, remove its entry from
`.vscode/settings.json`; VS Code will use the applicable user or default value.

## 6. Understand extensions, running, and debugging

Your installation already has support for Python, JavaScript/web development,
PHP, and C/C++, along with formatting and AI tools. You can inspect installed
extensions by opening Extensions and searching for `@installed`.

Add extensions when a project needs them. A language extension does not
necessarily install that language's runtime or compiler.

- **Run** executes a program using its required runtime or compiler.
- **Debug** lets you pause at breakpoints and inspect values while it runs.
- **Source Control** works with tools such as Git to track changes and history;
  saving a file and committing a Git change are different actions.
- **Problems** lists diagnostics reported by language tools. It does not
  guarantee that a program behaves correctly.

After you are comfortable with files and the terminal, pick one language and
set up a small program. That will make running, formatting, and debugging
concrete without needing to learn every extension at once.

## Official references

- [VS Code interface](https://code.visualstudio.com/docs/editing/getting-started/userinterface)
- [User and workspace settings](https://code.visualstudio.com/docs/configure/settings)
- [Tips and keyboard shortcuts](https://code.visualstudio.com/docs/editing/getting-started/tips-and-tricks)
