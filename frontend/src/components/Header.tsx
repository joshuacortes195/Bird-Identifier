import { BirdIcon } from "../icons";
import { ThemeToggle } from "./ThemeToggle";

export function Header() {
  return (
    <header className="mx-auto flex w-full max-w-3xl items-center justify-between px-5 pt-6 pb-2">
      <div className="flex items-center gap-3">
        <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-fg">
          <BirdIcon width={22} height={22} />
        </span>
        <div className="leading-tight">
          <h1 className="font-serif text-xl font-semibold tracking-tight">Bird Identifier</h1>
          <p className="text-xs text-muted">Fine-grained North American species</p>
        </div>
      </div>
      <ThemeToggle />
    </header>
  );
}
