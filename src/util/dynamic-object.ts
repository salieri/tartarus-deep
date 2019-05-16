import _ from 'lodash';
import Joi from 'joi';

export interface DynamicObject {
  [key: string]: any;
}


export default {

  attachParamSettersGetters(instance: DynamicObject, schema: object): void {
    _.each(
      schema,
      (validator: any, key: string) => {
        if (typeof (instance as any)[key] !== 'undefined') {
          throw new Error(`${instance.constructor.name} tries to overwrite existing method or property '${key}'`);
        }

        Object.defineProperty(
          instance,
          key,
          {
            value: (value?: any): any => {
              if (_.isUndefined(value) === true) {
                return instance.params[key];
              }

              const result = Joi.validate(value, validator);

              if (result.error) {
                throw result.error;
              }

              /* eslint-disable-next-line */
              instance.params[key] = result.value;

              return instance;
            },
          },
        );
      },
    );
  },


};
