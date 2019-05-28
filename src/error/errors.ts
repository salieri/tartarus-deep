
export class TartarusError extends Error {}

export class KeyNotFoundError extends TartarusError {
  public key: string;

  public constructor(message: string, key: string) {
    super(message);

    this.key = key;
  }
}


export class ValueNotSetError extends TartarusError {}
export class ValueNotDeclaredError extends TartarusError {}
export class InvalidValueError extends TartarusError {}

export class RecoverableCompilationError extends TartarusError {}
