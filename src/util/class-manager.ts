import _ from 'lodash';

// This prevents circular dependencies
export interface LayerLike {
  attachLayer: (layer: unknown) => void;
}

/* eslint-disable @typescript-eslint/no-explicit-any */

export interface ClassModule {
  [key: string]: any;
}

export interface InstanceParams {
  [key: string]: any;
}


export class ClassManager {
  protected knownClasses: ClassModule;

  protected baseClass: any;

  public constructor(knownClasses: ClassModule, baseClass: any) {
    this.baseClass = baseClass;
    this.knownClasses = this.prepareClasses(knownClasses);
  }


  protected prepareClasses(knownClasses: ClassModule): ClassModule {
    const filteredClasses: ClassModule = {};

    _.each(
      knownClasses,
      (cls, key) => {
        if (cls.prototype instanceof this.baseClass) {
          filteredClasses[ClassManager.getSimplifiedName(key)] = cls;
        }
      },
    );

    return filteredClasses;
  }


  public getKnownClasses(): ClassModule {
    return this.knownClasses;
  }


  public coerce(instanceDefinition: string | object, layer?: LayerLike, params?: InstanceParams): any {
    if (instanceDefinition instanceof this.baseClass) {
      return instanceDefinition;
    }

    if (_.isString(instanceDefinition) === false) {
      throw new Error('Cannot coerce: Invalid data');
    }

    return this.factory(instanceDefinition as string, layer, params);
  }


  public factory(className: string, layer?: LayerLike, params?: InstanceParams): any {
    // tslint:disable-next-line
    const ClassSpec = this.find(className);

    const instance = new ClassSpec(params);

    if (layer) {
      instance.attachLayer(layer);
    }

    return instance;
  }


  public find(className: string): any {
    const simplifiedName = ClassManager.getSimplifiedName(className);

    if (!(simplifiedName in this.knownClasses)) {
      throw new Error(`Unknown class: '${className}'`);
    }

    return this.knownClasses[simplifiedName];
  }


  public static getSimplifiedName(className: string): string {
    return className.toLowerCase().replace(/[^a-z0-9]/g, '');
  }
}

