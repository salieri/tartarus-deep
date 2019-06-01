import {
  Loggable,
  Logger,
  LogLevel,
} from './logger';


export class ConsoleLogger extends Logger {
  public log(loggable: Loggable): void {
    const out = JSON.stringify(loggable, null, 0);

    switch (loggable.level) {
      case LogLevel.Fatal:
      case LogLevel.Error:
        console.error(out);
        break;

      case LogLevel.Warn:
        console.warn(out);
        break;

      case LogLevel.Info:
        console.info(out);
        break;

      case LogLevel.Debug:
        console.log(out);
        break;

      case LogLevel.Trace:
        console.trace(out);
        break;

      default:
        console.log(out);
        break;
    }
  }
}
