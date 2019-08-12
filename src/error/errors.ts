
/*
 * @todo Find a better way of dealing with
 * @link https://github.com/Microsoft/TypeScript/wiki/Breaking-Changes#extending-built-ins-like-error-array-and-map-may-no-longer-work
 *
 */


export class TartarusError extends Error {
  public constructor(m: string) {
    super(m);

    Object.setPrototypeOf(this, TartarusError.prototype);
  }
}

export class KeyNotFoundError extends TartarusError {
  public key: string;

  public constructor(message: string, key: string) {
    super(message);

    this.key = key;

    Object.setPrototypeOf(this, KeyNotFoundError.prototype);
  }
}


export class ValueNotSetError extends TartarusError {
  public constructor(m: string) {
    super(m);

    Object.setPrototypeOf(this, ValueNotSetError.prototype);
  }
}


export class ValueNotDeclaredError extends TartarusError {
  public constructor(m: string) {
    super(m);

    Object.setPrototypeOf(this, ValueNotDeclaredError.prototype);
  }
}


export class InvalidValueError extends TartarusError {
  public constructor(m: string) {
    super(m);

    Object.setPrototypeOf(this, InvalidValueError.prototype);
  }
}


export class RecoverableCompilationError extends TartarusError {
  public constructor(m: string) {
    super(m);

    Object.setPrototypeOf(this, RecoverableCompilationError.prototype);
  }
}
