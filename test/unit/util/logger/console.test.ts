import { expect } from 'chai';
import { ConsoleLike, ConsoleLogger, LogLevel } from '../../../../src/util';

class CapturedConsole implements ConsoleLike {
  public history: { type: string; message?: any; optionalParams?: any[] }[] = [];

  public log(message?: any, ...optionalParams: any[]): void {
    this.history.push({ message, optionalParams, type: 'log' });
  }

  public info(message?: any, ...optionalParams: any[]): void {
    this.history.push({ message, optionalParams, type: 'info' });
  }

  public warn(message?: any, ...optionalParams: any[]): void {
    this.history.push({ message, optionalParams, type: 'warn' });
  }

  public error(message?: any, ...optionalParams: any[]): void {
    this.history.push({ message, optionalParams, type: 'error' });
  }

  public debug(message?: any, ...optionalParams: any[]): void {
    this.history.push({ message, optionalParams, type: 'debug' });
  }

  public trace(message?: any, ...optionalParams: any[]): void {
    this.history.push({ message, optionalParams, type: 'trace' });
  }
}

describe(
  'Console Logger',
  () => {
    it(
      'should log all types of console messages',
      () => {
        const c = new CapturedConsole();
        const l = new ConsoleLogger(c);

        l.info('MooInfo');
        l.warn('MooWarn');
        l.error('MooError');
        l.debug('MooDebug');
        l.trace('MooTrace');
        l.fatal('MooFatal');
        l.log({ type: 'MooLog', level: 200, time: new Date() });

        c.history[0].type.should.equal('info');
        JSON.parse(c.history[0].message).type.should.equal('MooInfo');

        c.history[1].type.should.equal('warn');
        JSON.parse(c.history[1].message).type.should.equal('MooWarn');

        c.history[2].type.should.equal('error');
        JSON.parse(c.history[2].message).type.should.equal('MooError');

        c.history[3].type.should.equal('log');
        JSON.parse(c.history[3].message).type.should.equal('MooDebug');

        c.history[4].type.should.equal('trace');
        JSON.parse(c.history[4].message).type.should.equal('MooTrace');

        c.history[5].type.should.equal('error');
        JSON.parse(c.history[5].message).type.should.equal('MooFatal');

        c.history[6].type.should.equal('log');
        JSON.parse(c.history[6].message).type.should.equal('MooLog');
      },
    );


    it(
      'should call a logger function to resolve what to log',
      () => {
        const c = new CapturedConsole();
        const l = new ConsoleLogger(c);

        l.info('moo', () => 'hello world');

        const c1 = JSON.parse(c.history[0].message);

        c1.type.should.equal('moo');
        c1.level.should.equal(LogLevel.Info);
        c1.message.should.equal('hello world');
      },
    );


    it(
      'should log all types of data',
      () => {
        const c = new CapturedConsole();
        const l = new ConsoleLogger(c);

        l.info('moo', true, 12.34, { moo: 'test' }, [1, 2, 3], ['hello', 'world'], [true, false], () => ['ouch', 'ouch2'], null);

        const c1 = JSON.parse(c.history[0].message);

        c1.type.should.equal('moo');
        c1.level.should.equal(LogLevel.Info);

        c1.value.should.equal(true);
        c1.value2.should.equal(12.34);
        c1.moo.should.equal('test');
        c1.value4.should.deep.equal([1, 2, 3]);
        c1.value5.should.deep.equal(['hello', 'world']);
        c1.value6.should.deep.equal([true, false]);
        c1.value7.should.deep.equal(['ouch', 'ouch2']);

        expect(c1.value8).to.equal(null);
      },
    );


    it(
      'should not call a logger function, if message is filtered',
      () => {
        const c = new CapturedConsole();
        const l = new ConsoleLogger(c);

        let called = false;

        // @ts-ignore
        l.logLevelConfig = {
          '*': LogLevel.Error,
        };

        l.info('moo', () => { called = true; return 'hello world'; });

        c.history.length.should.equal(0);

        called.should.equal(false);
      },
    );


    it(
      'should log anything, if no log level config exists',
      () => {
        const c = new CapturedConsole();
        const l = new ConsoleLogger(c);

        // @ts-ignore
        l.logLevelConfig = {};

        l.debug('hello', 'world');
        l.info('world', 'hello');

        JSON.parse(c.history[0].message).message.should.equal('world');
        JSON.parse(c.history[1].message).message.should.equal('hello');
      },
    );


    it(
      'should filter what it logs depending on the log level config',
      () => {
        const c = new CapturedConsole();
        const l = new ConsoleLogger(c);

        // @ts-ignore
        l.logLevelConfig = {
          '*': LogLevel.Error,
          moo: LogLevel.Warn,
        };

        l.debug('hello.world', 'message goes here');
        l.debug('moo', 'message goes here');
        l.warn('moo', 'warning!');
        l.error('hello.another', 'error!');

        c.history.length.should.equal(2);

        c.history[0].type.should.equal('warn');
        c.history[1].type.should.equal('error');

        const c1 = JSON.parse(c.history[0].message);

        c1.message.should.equal('warning!');

        const c2 = JSON.parse(c.history[1].message);

        c2.message.should.equal('error!');
      },
    );
  },
);
