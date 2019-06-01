import { Logger } from './logger';

export class MuteLogger extends Logger {
  public log(): void {
    // do nothing
  }
}
