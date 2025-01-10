class TimeoutCancelled { constructor(public name: string) { } };

export class Timeout {
  public cancelled: TimeoutCancelled;

  private token?: number;
  private reject?: (t: TimeoutCancelled) => void;
  public done: boolean = false;

  constructor(private duration: number, public name: string) {
    this.cancelled = new TimeoutCancelled(name);
  }

  private promise?: Promise<void>
  public start(): Promise<void> {
    if (this.duration === 0) {
      return new Promise((resolve) => resolve());
    }

    if (this.promise) return this.promise;

    return this.promise = new Promise((resolve, reject) => {
      this.token = setTimeout(() => {
        this.done = true;
        resolve();
      }, this.duration);
      this.reject = reject;
    })
  }

  public cancel() {
    if (this.done)
      throw "Attempted to cancel a timeout that finished"

    clearTimeout(this.token)
    if (this.reject) this.reject(this.cancelled);
  }
}
