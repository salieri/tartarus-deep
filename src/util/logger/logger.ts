import _ from 'lodash';

export interface Loggable {
  type: string;
  level: number;
  time: Date;
}

export interface LogObject {
  [key: string]: any;
}

export type LogData = string|number|boolean|undefined|null|LogObject|LogObject[]|string[]|number[]|boolean[]|Function;


export enum LogLevel {
  Trace = 10,
  Debug = 20,
  Info = 30,
  Warn = 40,
  Error = 50,
  Fatal = 60,
}


export interface LogLevelConfig {
  [key: string]: LogLevel;
}


export abstract class Logger {
  /**
   * Simple wildcard matching at the end of the string (e.g. model.compile.*) is supported, nothing else
   */
  protected logLevelConfig: LogLevelConfig = {
    '*': LogLevel.Trace,
  };

  public fatal(type: string, ...logData: LogData[]): void {
    this.filter(type, LogLevel.Fatal, ...logData);
  }


  public error(type: string, ...logData: LogData[]): void {
    this.filter(type, LogLevel.Error, ...logData);
  }


  public warn(type: string, ...logData: LogData[]): void {
    this.filter(type, LogLevel.Warn, ...logData);
  }


  public info(type: string, ...logData: LogData[]): void {
    this.filter(type, LogLevel.Info, ...logData);
  }


  public debug(type: string, ...logData: LogData[]): void {
    this.filter(type, LogLevel.Debug, ...logData);
  }


  public trace(type: string, ...logData: LogData[]): void {
    this.filter(type, LogLevel.Trace, ...logData);
  }


  protected shouldLog(type: string, logLevel: LogLevel): boolean {
    const match = _.reduce(
      this.logLevelConfig,
      (accumulator: string, configLogLevel: LogLevel, key: string): string => {
        const exact = (key.substr(key.length - 1) !== '*');

        if ((exact) && (key === type)) {
          return key;
        }

        if (
          (!exact)
          && (type.substr(0, key.length - 1) === key.substr(0, key.length - 1))
          && (key.length > accumulator.length)
          && ((accumulator === '') || (accumulator.substr(accumulator.length - 1) === '*'))
        ) {
          return key;
        }

        return accumulator;
      },
      '',
    );

    if (match === '') {
      return true;
    }

    return (logLevel >= this.logLevelConfig[match]);
  }


  protected filter(type: string, logLevel: number, ...logData: LogData[]): void {
    if (!this.shouldLog(type, logLevel)) {
      return;
    }

    this.log(this.coerce(type, logLevel, ...logData));
  }


  protected coerceObject(logData: LogData, index: number): LogObject {
    const ld = _.isFunction(logData) ? logData() : logData;

    const result: LogObject = {};

    const ordinal = index > 0 ? `${(index + 1)}` : '';

    // eslint-disable-next-line default-case
    switch (typeof ld) {
      case 'string':
        result[`message${ordinal}`] = ld;
        break;

      case 'number':
      case 'bigint':
      case 'boolean':
      case 'undefined':
        result[`value${ordinal}`] = ld;
    }

    if (_.isArray(ld)) {
      result[`value${ordinal}`] = ld;
    }

    if (ld === null) {
      result[`value${ordinal}`] = ld;
    }

    return (_.keys(result).length > 0) ? result : ld as object;
  }


  public coerce(type: string, level: LogLevel, ...logData: LogData[]): Loggable {
    const baseObject = {
      type,
    };

    const extendedBaseObject = {
      type,
      level,
      time: new Date(),
    };

    return _.merge(
      baseObject,
      ...(_.map(logData, (ld: LogData, index: number) => this.coerceObject(ld, index))),
      extendedBaseObject, // 'type' is used twice on purpose -- this way it gets first order in the object and will not be overwritten
    );
  }


  public abstract log(loggable: Loggable): void;
}
