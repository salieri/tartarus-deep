import _ from 'lodash';

import {
  Loggable,
  Logger,
  LogData,
  LogLevel,
} from './logger';


export class ContextLogger extends Logger {
  private context: LogData;

  private logger: Logger;

  public constructor(logger: Logger, context: LogData) {
    super();

    this.logger = logger;
    this.context = this.determineContext(context);
  }


  public coerce(type: string, level: LogLevel, ...logData: LogData[]): Loggable {
    return _.merge(
      this.logger.coerce(type, level, ...logData),
      {
        context: this.context,
      },
    );
  }


  protected shouldLog(type: string, logLevel: LogLevel): boolean {
    // @ts-ignore TS2446
    return this.logger.shouldLog(type, logLevel);
  }


  public log(loggable: Loggable): void {
    this.logger.log(loggable);
  }


  public getContext(): LogData {
    return this.context;
  }


  protected determineContext(context: LogData): LogData {
    if ((this.logger as any).getContext) {
      return [
        ..._.castArray((this.logger as any).getContext()),
        context,
      ];
    }

    return context;
  }
}
