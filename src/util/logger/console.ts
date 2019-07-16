import {
  Loggable,
  Logger,
  LogLevel,
} from './logger';

export interface ConsoleLike {
  error(message?: any, ...optionalParams: any[]): void;
  warn(message?: any, ...optionalParams: any[]): void;
  info(message?: any, ...optionalParams: any[]): void;
  log(message?: any, ...optionalParams: any[]): void;
  trace(message?: any, ...optionalParams: any[]): void;
}


export class ConsoleLogger extends Logger {
  private console: ConsoleLike;

  public constructor(consoleOverride: ConsoleLike = console) {
    super();

    this.console = consoleOverride;
  }

  public log(loggable: Loggable): void {
    const out = JSON.stringify(loggable, null, 0);

    switch (loggable.level) {
      case LogLevel.Fatal:
      case LogLevel.Error:
        this.console.error(out);
        break;

      case LogLevel.Warn:
        this.console.warn(out);
        break;

      case LogLevel.Info:
        this.console.info(out);
        break;

      case LogLevel.Debug:
        this.console.log(out);
        break;

      case LogLevel.Trace:
        this.console.trace(out);
        break;

      default:
        this.console.log(out);
        break;
    }
  }
}
